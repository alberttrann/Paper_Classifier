# streamlit_app.py

import streamlit as st
import os
import re
import string
import numpy as np
import pandas as pd
import joblib
import torch
import altair as alt
from collections import Counter

# NLTK
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Sentence Transformers
from sentence_transformers import SentenceTransformer

# Sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import hstack, csr_matrix

# Transformers (for the fine-tuned model)
from transformers import AutoTokenizer, AutoModel

# --- Page Configuration ---
st.set_page_config(
    page_title="arXiv Classification Engine",
    page_icon="🔬",
    layout="wide"
)

# --- Text Preprocessing Logic (Must be identical to training) ---
@st.cache_resource
def load_nltk_data():
    try:
        stopwords.words('english')
    except LookupError:
        import nltk
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

load_nltk_data()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
domain_specific_stopwords = {'result', 'study', 'show', 'paper', 'model', 'analysis', 'method', 'approach', 'propose', 'demonstrate', 'investigate', 'present', 'based', 'using', 'also', 'however', 'provide', 'describe'}
stop_words.update(domain_specific_stopwords)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(cleaned_tokens)

# --- Custom Transformer Model Definition (for fine-tuned e5) ---
class MultiLabelTransformer(torch.nn.Module):
    def __init__(self, base_model_path, n_classes):
        super(MultiLabelTransformer, self).__init__()
        self.transformer = AutoModel.from_pretrained(base_model_path)
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(self.transformer.config.hidden_size, n_classes)
    def forward(self, input_ids, attention_mask):
        transformer_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = transformer_output.pooler_output
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        return logits

# --- Model Loading (with Streamlit Caching for Performance) ---
@st.cache_resource
def load_all_production_models():
    models = {}
    prod_path = "./production_models"
    ft_path = "./e5_finetuned_multilabel"
    
    # --- Load all saved components ---
    st.info("Loading models... This may take a moment on first run.")
    
    # 1. Load Universal Components
    models['labels'] = joblib.load(os.path.join(prod_path, 'final_labels.pkl'))
    models['tfidf_vectorizer'] = joblib.load(os.path.join(prod_path, 'tfidf_vectorizer.pkl'))
    models['sbert_model'] = SentenceTransformer(os.path.join(prod_path, 'e5_embedding_model'))

    # 2. Load Single-Label Models
    models['model_lr_tfidf'] = joblib.load(os.path.join(prod_path, 'model_lr_tfidf.pkl'))
    models['base_mnb'] = joblib.load(os.path.join(prod_path, 'base_model_mnb.pkl'))
    models['base_knn_calibrated'] = joblib.load(os.path.join(prod_path, 'base_model_knn_calibrated.pkl'))
    models['base_dt_calibrated'] = joblib.load(os.path.join(prod_path, 'base_model_dt_calibrated.pkl'))
    models['stacking_meta_learner'] = joblib.load(os.path.join(prod_path, 'stacking_meta_learner.pkl'))

    # 3. Load Fine-Tuned e5 Model
    models['ft_tokenizer'] = AutoTokenizer.from_pretrained(ft_path)
    n_classes = len(models['labels'])
    ft_model = MultiLabelTransformer(ft_path, n_classes=n_classes)
    classifier_weights_path = os.path.join(ft_path, "classifier_weights.bin")
    ft_model.classifier.load_state_dict(torch.load(classifier_weights_path, map_location=torch.device('cpu')))
    models['ft_e5_model'] = ft_model.eval()

    # Create ML Binarizer for multi-label
    models['mlb'] = MultiLabelBinarizer(classes=models['labels'])
    models['mlb'].fit(models['labels']) # Fit the binarizer

    st.success("All production models loaded!")
    return models

# --- Prediction Functions (Accurate Implementations) ---

def run_single_label_pipeline(abstract, models):
    labels = models['labels']
    results = {}
    logs = {}

    # 1. Preprocess & Feature Engineering
    log_1 = "1. Cleaning text (lemmatization, custom stop words...)\n"
    cleaned_abstract = clean_text(abstract)
    log_2 = f"2. Generating features:\n   - Advanced TF-IDF (1,2-ngrams, min_df=5...)\n   - e5-base Semantic Embedding (768 dims)\n"
    tfidf_features = models['tfidf_vectorizer'].transform([cleaned_abstract])
    embedding_features = models['sbert_model'].encode([cleaned_abstract])
    
    # --- Model 1: Fine-tuned e5 (Single-Label mode) ---
    log_3 = "--- Model 1: Fine-Tuned e5 ---\n"
    log_3 += "Running Transformer forward pass...\n"
    inputs = models['ft_tokenizer'].encode_plus(cleaned_abstract, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
    with torch.no_grad():
        logits = models['ft_e5_model'](inputs['input_ids'], inputs['attention_mask'])
    ft_probs = torch.softmax(logits, dim=1).flatten().numpy()
    ft_pred_idx = np.argmax(ft_probs)
    results['Fine-Tuned e5 (Champion)'] = {'prediction': labels[ft_pred_idx], 'probabilities': ft_probs}
    log_3 += f"Applying Softmax -> Predicted Index: {ft_pred_idx} ({labels[ft_pred_idx]}) with confidence {ft_probs[ft_pred_idx]:.2%}\n"
    logs['ft_e5'] = log_3

    # --- Model 2: Best Single Model (LR on TFIDF) ---
    log_4 = "--- Model 2: Best Single (LR(tfidf)) ---\n"
    lr_probs = models['model_lr_tfidf'].predict_proba(tfidf_features)[0]
    lr_pred_idx = np.argmax(lr_probs)
    results['Best Single Model (LR on TFIDF)'] = {'prediction': labels[lr_pred_idx], 'probabilities': lr_probs}
    log_4 += f"Running LR.predict_proba on TF-IDF features...\nPredicted Index: {lr_pred_idx} ({labels[lr_pred_idx]}) with confidence {lr_probs[lr_pred_idx]:.2%}\n"
    logs['lr_tfidf'] = log_4
    
    # --- Base Model Predictions (for Ensembles) ---
    base_logs = "--- Generating Base Predictions for Ensembles ---\n"
    mnb_probs = models['base_mnb'].predict_proba(tfidf_features)[0]
    base_logs += f"1. Tuned MNB(tfidf) prob dist: [Shape: {mnb_probs.shape}]\n"
    knn_calibrated_probs = models['base_knn_calibrated'].predict_proba(embedding_features)[0]
    base_logs += f"2. Calibrated kNN(emb) prob dist: [Shape: {knn_calibrated_probs.shape}]\n"
    dt_calibrated_probs = models['base_dt_calibrated'].predict_proba(tfidf_features)[0]
    base_logs += f"3. Calibrated DT(tfidf) prob dist: [Shape: {dt_calibrated_probs.shape}]\n"

    # --- Model 3: Soft Voting Ensemble ---
    log_5 = "--- Model 3: Soft Voting Ensemble ---\n"
    log_5 += base_logs
    soft_vote_probs = (0.4 * mnb_probs) + (0.4 * knn_calibrated_probs) + (0.2 * dt_calibrated_probs)
    soft_vote_pred_idx = np.argmax(soft_vote_probs)
    results['Soft Voting Ensemble'] = {'prediction': labels[soft_vote_pred_idx], 'probabilities': soft_vote_probs}
    log_5 += f"Averaging probabilities with weights (40/40/20)...\nPredicted Index: {soft_vote_pred_idx} ({labels[soft_vote_pred_idx]}) with confidence {soft_vote_probs[soft_vote_pred_idx]:.2%}\n"
    logs['soft_vote'] = log_5
    
    # --- Model 4: Top Stacking Ensemble ---
    log_6 = "--- Model 4: Stacking Champion ---\n"
    log_6 += base_logs
    # Re-create meta-features for stacking (using calibrated probs for inference is fine)
    meta_features_inference = np.hstack([mnb_probs.reshape(1,-1), knn_calibrated_probs.reshape(1,-1), dt_calibrated_probs.reshape(1,-1)])
    log_6 += f"Creating L1 feature vector from base probs: [Shape: {meta_features_inference.shape}]\n"
    # Create final L1 input by stacking with original TF-IDF
    meta_learner_input = hstack([csr_matrix(meta_features_inference), tfidf_features]).tocsr()
    log_6 += f"Final meta-learner input vector: [Shape: {meta_learner_input.shape}]\n"
    stacking_probs = models['stacking_meta_learner'].predict_proba(meta_learner_input)[0]
    stacking_pred_idx = np.argmax(stacking_probs)
    results['Top Stacking Ensemble'] = {'prediction': labels[stacking_pred_idx], 'probabilities': stacking_probs}
    log_6 += f"Running Meta-Learner (LR) on vector...\nPredicted Index: {stacking_pred_idx} ({labels[stacking_pred_idx]}) with confidence {stacking_probs[stacking_pred_idx]:.2%}\n"
    logs['stacking'] = log_6
    
    return results, labels, logs, (cleaned_abstract, tfidf_features.shape, embedding_features.shape)

def run_multi_label_pipeline(abstract, models):
    labels = models['labels']
    ft_e5 = models['ft_e5_model']
    ft_tokenizer = models['ft_tokenizer']
    logs = {}

    log_1 = "1. Cleaning text (lemmatization, custom stop words...)\n"
    cleaned_abstract = clean_text(abstract)
    log_1 += f"2. Tokenizing text for {ft_e5.transformer.name_or_path}...\n"
    inputs = ft_tokenizer.encode_plus(
        cleaned_abstract, return_tensors='pt', padding='max_length',
        truncation=True, max_length=512
    )
    log_1 += f"3. Running Transformer forward pass... (Input shape: {inputs['input_ids'].shape})\n"
    with torch.no_grad():
        logits = ft_e5(inputs['input_ids'], inputs['attention_mask'])
    
    log_1 += f"4. Applying independent Sigmoid to 8 output neurons...\n"
    ft_probs = torch.sigmoid(logits).flatten().numpy()
    
    threshold = 0.5
    pred_indices = np.where(ft_probs >= threshold)[0]
    predictions = [labels[i] for i in pred_indices]
    log_1 += f"5. Applying 0.5 threshold. Found {len(predictions)} labels: {predictions}\n"

    results = {
        'Fine-Tuned e5 (Multi-Label)': {
            'predictions': predictions if len(predictions) > 0 else ["No category above threshold"],
            'probabilities': ft_probs
        }
    }
    return results, labels, log_1

# --- Streamlit App UI ---
st.title("🔬 arXiv Abstract Classification Engine")
st.markdown("An interactive demo of champion models from a systematic benchmarking project. This app runs the **actual trained production models** to classify scientific abstracts in real-time.")

# --- Load all models on startup ---
with st.spinner("Warming up the models... This may take a minute on first load."):
    try:
        models = load_all_production_models()
    except FileNotFoundError:
        st.error(f"FATAL ERROR: Could not find model files. Make sure the '{OUTPUT_DIR}' and 'e5_finetuned_multilabel' directories exist and are populated. Run 'train_and_save_production_models.py' first.")
        st.stop()

st.sidebar.title("About this Project")
st.sidebar.info("""
This project systematically compares text classification architectures, from single models to complex ensembles.
- **Single-Label:** Trained on 40,000 single-label abstracts across 8 categories.
- **Multi-Label:** Trained on 77,000+ multi-label abstracts.

This demo runs the **fully-trained, optimized models** to showcase the final results.
""")
st.sidebar.markdown(f"**Champion Models Loaded:**\n"
                    f"* Fine-Tuned e5 Transformer\n"
                    f"* Stacking Ensemble (7 components)\n"
                    f"* Soft Voting Ensemble (3 components)\n"
                    f"* Best Single LR(tfidf)\n"
                    f"* All required base models & vectorizers")

# --- TABS ---
tab1, tab2 = st.tabs(["Single-Label Classification", "Multi-Label Classification"])

# --- TAB 1: SINGLE-LABEL ---
with tab1:
    st.header("Single-Label Classification")
    st.markdown("These models predict the **single best category** for an abstract. Each model represents a champion from our different experimental phases.")
    
    example_abstract_sl = "We present a study of the star formation history in the Large Magellanic Cloud. Using deep photometric data from the Hubble Space Telescope, we analyze the color-magnitude diagram of stellar populations. Our results indicate a burst of star formation approximately 2 billion years ago. We compare these findings with existing cosmological models and discuss the implications for galaxy evolution."
    abstract_input_sl = st.text_area("Enter a scientific abstract here:", value=example_abstract_sl, height=250, key="sl_abstract")
    
    if st.button("Classify Single-Label", key="sl_button"):
        if not abstract_input_sl.strip():
            st.error("Please enter an abstract.")
        else:
            sl_results, sl_labels, logs, feature_shapes = run_single_label_pipeline(abstract_input_sl, models)
            
            st.success("Classification Complete! Here are the predictions from our champion models:")
            
            cols = st.columns(len(sl_results))
            for idx, (model_name, result) in enumerate(sl_results.items()):
                with cols[idx]:
                    st.metric(label=model_name, value=result['prediction'])
            
            st.markdown("---")
            st.subheader("Visualizing the Model Decisions")
            
            # Prepare data for visualization
            viz_data = []
            for model_name, result in sl_results.items():
                for i, prob in enumerate(result['probabilities']):
                    viz_data.append({'Model': model_name.split(' (')[0], 'Category': sl_labels[i], 'Probability': prob})
            
            df_viz = pd.DataFrame(viz_data)
            
            chart = alt.Chart(df_viz).mark_bar().encode(
                x=alt.X('Probability:Q', axis=alt.Axis(format='.0%')),
                y=alt.Y('Category:N', sort='-x'),
                color=alt.Color('Model:N', legend=alt.Legend(title="Model")),
                row=alt.Row('Model:N', header=alt.Header(title="Model Decision Process", labels=False), sort=alt.EncodingSortField("Probability", op="max", order='descending')),
                tooltip=['Model', 'Category', 'Probability']
            ).properties(
                title='Model Confidence Scores per Category'
            ).interactive()
            
            st.altair_chart(chart, use_container_width=True)
            
            st.subheader("Behind the Scenes: The Inference Logs")
            st.markdown("This shows the exact operations happening for each model, demonstrating the different logic and feature pipelines.")
            
            l_col, r_col = st.columns(2)
            with l_col:
                with st.expander("Feature Pipeline Log"):
                    st.text(f"Raw Abstract -> clean_text(...)\n-> Processed Text: '{feature_shapes[0][:50]}...'\n")
                    st.text(f"-> TfidfVectorizer -> Sparse Matrix [Shape: {feature_shapes[1]}]")
                    st.text(f"-> e5-base Encoder -> Dense Matrix [Shape: {feature_shapes[2]}]")
                
                with st.expander("Soft Voting Log"):
                    st.text(logs['soft_vote'])
                
                with st.expander("Best Single Model (LR) Log"):
                    st.text(logs['lr_tfidf'])
            
            with r_col:
                with st.expander("Fine-Tuned e5 Log"):
                    st.text(logs['ft_e5'])
                    
                with st.expander("Stacking Champion Log"):
                    st.text(logs['stacking'])

# --- TAB 2: MULTI-LABEL ---
with tab2:
    st.header("Multi-Label Classification")
    st.markdown("This model predicts **all relevant categories** for an abstract. It uses our fine-tuned e5 Transformer, which was trained on 77,000+ multi-label examples.")
    
    example_abstract_ml = "We propose a novel algorithm for graph-based semi-supervised learning that leverages principles from statistical mechanics. Our method models label propagation as a diffusion process on a weighted graph, where the weights are determined by a Hamiltonian derived from quantum field theory. We demonstrate that this approach significantly outperforms existing methods on benchmark datasets, particularly in low-label regimes. The computational complexity is analyzed using methods from theoretical computer science."
    abstract_input_ml = st.text_area("Enter a scientific abstract here:", value=example_abstract_ml, height=250, key="ml_abstract")

    if st.button("Classify Multi-Label", key="ml_button"):
        if not abstract_input_ml.strip():
            st.error("Please enter an abstract.")
        else:
            ml_results, ml_labels, ml_log = run_multi_label_pipeline(abstract_input_ml, models)

            st.success("Classification Complete!")
            
            result = ml_results['Fine-Tuned e5 (Multi-Label)']
            
            st.subheader("Predicted Categories:")
            # Display predictions as nice tags
            tags_html = "".join([f"<span style='background-color: #0072B2; color: white; padding: 5px 10px; border-radius: 15px; margin: 3px;'>{tag}</span>" for tag in result['predictions']])
            st.markdown(tags_html, unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("Underlying Process: Per-Category Confidence Scores")
            st.markdown("This chart shows the independent probability for each category. Any category with a score above the 50% threshold (red line) is included in the final prediction.")
            
            # Prepare data for visualization
            viz_data_ml = []
            for i, prob in enumerate(result['probabilities']):
                viz_data_ml.append({'Category': ml_labels[i], 'Probability': prob})
            
            df_viz_ml = pd.DataFrame(viz_data_ml)
            
            bar_chart = alt.Chart(df_viz_ml).mark_bar().encode(
                x=alt.X('Probability:Q', axis=alt.Axis(format='.0%')),
                y=alt.Y('Category:N', sort='-x'),
                color=alt.condition(
                    alt.datum.Probability > 0.5,
                    alt.value('orange'),  # Color for bars above threshold
                    alt.value('steelblue') # Color for bars below threshold
                ),
                tooltip=['Category', 'Probability']
            ).properties(
                title='Fine-Tuned Model Confidence per Category'
            )
            
            rule = alt.Chart(pd.DataFrame({'threshold': [0.5]})).mark_rule(color='red', strokeWidth=2, strokeDash=[5, 5]).encode(x='threshold:Q')
            
            st.altair_chart(bar_chart + rule, use_container_width=True)

            with st.expander("Behind the Scenes: The Inference Log"):
                st.text(ml_log)