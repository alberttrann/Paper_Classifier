# run_champion_pipeline.py

import os
import re
import string
import time
import numpy as np
from datetime import datetime
from datasets import load_dataset
from collections import Counter

# NLTK for text cleaning
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Transformers for loading SciBERT correctly
from transformers import AutoTokenizer, AutoModel

# Scikit-learn for models, vectorizers, and metrics
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report

# XGBoost for a powerful meta-learner
import xgboost as xgb

# Other utilities
from scipy.sparse import hstack, issparse
from tqdm.auto import tqdm
import torch

# --- Configuration ---
# Data Sampling
CATEGORIES_TO_SELECT = ['astro-ph', 'cond-mat', 'cs', 'math', 'physics']
SAMPLES_PER_CATEGORY = 2000 # Using 2000 for a robust result (10k total)

# Models & Vectorizers
SCIBERT_MODEL_NAME = "allenai/scibert_scivocab_uncased"
TFIDF_MAX_FEATURES = 10000
RANDOM_STATE = 42
CV_FOLDS = 5 # Use 3 folds for faster GridSearchCV. Increase to 5 for more robust tuning.
BATCH_SIZE = 128 # SciBERT can be more memory intensive, start with a smaller batch size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_FILE_PATH = "benchmark_results.txt"

# --- NLTK Downloads ---
# (Assuming they are already downloaded)

# --- Helper function for logging ---
def log_message(message, to_console=True):
    if to_console:
        print(message)
    with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

# --- Enhanced Text Preprocessing Function ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
# Add custom, domain-specific stop words
domain_specific_stopwords = {
    'result', 'study', 'show', 'paper', 'model', 'analysis', 'method', 
    'approach', 'propose', 'demonstrate', 'investigate', 'present', 
    'based', 'using', 'also', 'however', 'propose', 'provide', 'describe'
}
stop_words.update(domain_specific_stopwords)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(cleaned_tokens)

# --- SciBERT Embedding Function ---
def get_scibert_embeddings(texts, model, tokenizer, device, batch_size=32):
    """Generates sentence embeddings from a base Transformer model using mean pooling."""
    all_embeddings = []
    model.eval()
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating SciBERT Embeddings"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Mean pooling: average the token embeddings, respecting the attention mask
        attention_mask = inputs['attention_mask']
        last_hidden_state = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        batch_embeddings = sum_embeddings / sum_mask
        all_embeddings.append(batch_embeddings.cpu().numpy())
    return np.vstack(all_embeddings)

# --- Main Execution ---
log_message("\n\n" + "="*80)
log_message(f"--- Champion Pipeline Benchmark Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
log_message("="*80)

# 1. Data Sampling and Preprocessing
print("--- Step 1: Data Sampling & Preprocessing ---")
category_counts = {cat: 0 for cat in CATEGORIES_TO_SELECT}
samples = []
dataset_generator = load_dataset("UniverseTBD/arxiv-abstracts-large", split="train", streaming=True)
for s in tqdm(dataset_generator, desc="Scanning for samples"):
    if all(count >= SAMPLES_PER_CATEGORY for count in category_counts.values()):
        break
    if s['categories'] is None or s['abstract'] is None or len(s['categories'].split(' ')) != 1:
        continue
    parent_category = s['categories'].strip().split('.')[0]
    if parent_category in CATEGORIES_TO_SELECT and category_counts[parent_category] < SAMPLES_PER_CATEGORY:
        s['parent_category'] = parent_category
        samples.append(s)
        category_counts[parent_category] += 1
print(f"Finished sampling. Total samples collected: {len(samples)}")
abstracts = [sample['abstract'] for sample in samples]
labels_str = [sample['parent_category'] for sample in samples]
processed_abstracts = [clean_text(abstract) for abstract in tqdm(abstracts, desc="Cleaning Abstracts")]
unique_labels = sorted(list(set(labels_str)))
label_to_int = {label: i for i, label in enumerate(unique_labels)}
labels = np.array([label_to_int[label] for label in labels_str])
train_texts, test_texts, y_train, y_test = train_test_split(
    processed_abstracts, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
)

# 2. Feature Engineering
print("\n--- Step 2: Enhanced Feature Engineering ---")
# Advanced TF-IDF
print("Creating Enhanced TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES, min_df=5, max_df=0.7, 
    sublinear_tf=True, ngram_range=(1, 2)
)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
X_test_tfidf = tfidf_vectorizer.transform(test_texts)
# Bag of Words (for meta-learner experiments)
print("Creating Bag of Words features...")
bow_vectorizer = CountVectorizer(vocabulary=tfidf_vectorizer.vocabulary_)
X_train_bow = bow_vectorizer.fit_transform(train_texts)
X_test_bow = bow_vectorizer.transform(test_texts)
# SciBERT Embeddings
print(f"Loading SciBERT model and tokenizer: {SCIBERT_MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(SCIBERT_MODEL_NAME)
scibert_model = AutoModel.from_pretrained(SCIBERT_MODEL_NAME, torch_dtype="auto").to(DEVICE)
X_train_emb = get_scibert_embeddings(train_texts, scibert_model, tokenizer, DEVICE, BATCH_SIZE)
X_test_emb = get_scibert_embeddings(test_texts, scibert_model, tokenizer, DEVICE, BATCH_SIZE)
print("All feature sets created.")

# 3. Hyperparameter Tuning of Base Models
print("\n--- Step 3: Hyperparameter Tuning of Base Models ---")
# MNB
print("Tuning MultinomialNB...")
mnb_params = {'alpha': [0.01, 0.1, 0.5, 1.0]}
mnb_grid = GridSearchCV(MultinomialNB(), mnb_params, cv=CV_FOLDS, n_jobs=-1)
mnb_grid.fit(X_train_tfidf, y_train)
best_mnb = mnb_grid.best_estimator_
log_message(f"Best MNB params: {mnb_grid.best_params_}", to_console=True)
# kNN
print("Tuning KNeighborsClassifier...")
knn_params = {'n_neighbors': [5, 7, 11], 'weights': ['uniform', 'distance']}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=CV_FOLDS, n_jobs=-1)
knn_grid.fit(X_train_emb, y_train)
best_knn = knn_grid.best_estimator_
log_message(f"Best kNN params: {knn_grid.best_params_}", to_console=True)
# DT
print("Tuning DecisionTreeClassifier...")
dt_params = {'max_depth': [15, 20, 30], 'min_samples_leaf': [1, 5, 10]}
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_STATE), dt_params, cv=CV_FOLDS, n_jobs=-1)
dt_grid.fit(X_train_tfidf, y_train)
best_dt = dt_grid.best_estimator_
log_message(f"Best DT params: {dt_grid.best_params_}", to_console=True)

# 4. Generate Out-of-Fold Predictions for Stacking
print("\n--- Step 4: Calibrating and Generating Meta-Features ---")
calibrated_dt = CalibratedClassifierCV(estimator=best_dt, cv=CV_FOLDS, method='isotonic')
calibrated_knn = CalibratedClassifierCV(estimator=best_knn, cv=CV_FOLDS, method='isotonic')
base_models_for_stacking = {'MNB_tfidf': best_mnb, 'kNN_emb': calibrated_knn, 'DT_tfidf': calibrated_dt}
meta_features_train = {}
for name, model in base_models_for_stacking.items():
    feature_type = name.split('_')[1]
    print(f"Generating out-of-fold predictions for {name}...")
    X_for_model = X_train_tfidf if feature_type == 'tfidf' else X_train_emb
    meta_features_train[name] = cross_val_predict(model, X_for_model, y_train, cv=CV_FOLDS, method='predict_proba', n_jobs=-1)
# Train the calibrated models on the full training data
calibrated_dt.fit(X_train_tfidf, y_train)
calibrated_knn.fit(X_train_emb, y_train)
print("All meta-features for stacking generated.")

# 5. Train Base Models on Full Training Data
print("\n--- Step 5: Training Base Models for Final Predictions ---")
# The tuned MNB and calibrated models are already fitted on the full training data
test_predictions = {
    'MNB_tfidf': best_mnb.predict_proba(X_test_tfidf),
    'kNN_emb': calibrated_knn.predict_proba(X_test_emb),
    'DT_tfidf': calibrated_dt.predict_proba(X_test_tfidf)
}
print("Base model predictions on test set generated.")

# 6. Define and Benchmark Meta-Learner Configurations
print("\n--- Step 6: Benchmarking Stacking Meta-Learner Configurations ---")
meta_learner_configs = {
    "LR(TFIDF)":  {'learner': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000), 'X_train': X_train_tfidf, 'X_test': X_test_tfidf},
    "LR(BoW)":    {'learner': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000), 'X_train': X_train_bow, 'X_test': X_test_bow},
    "LR(Emb)":    {'learner': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000), 'X_train': X_train_emb, 'X_test': X_test_emb},
    "XGB(TFIDF)": {'learner': xgb.XGBClassifier(random_state=RANDOM_STATE), 'X_train': X_train_tfidf, 'X_test': X_test_tfidf},
    "XGB(BoW)":   {'learner': xgb.XGBClassifier(random_state=RANDOM_STATE), 'X_train': X_train_bow, 'X_test': X_test_bow},
    "XGB(Emb)":   {'learner': xgb.XGBClassifier(random_state=RANDOM_STATE), 'X_train': X_train_emb, 'X_test': X_test_emb},
    "GNB(TFIDF)": {'learner': GaussianNB(), 'X_train': X_train_tfidf.toarray(), 'X_test': X_test_tfidf.toarray()}, # GNB requires dense
    "GNB(BoW)":   {'learner': GaussianNB(), 'X_train': X_train_bow.toarray(), 'X_test': X_test_bow.toarray()},
    "GNB(Emb)":   {'learner': GaussianNB(), 'X_train': X_train_emb, 'X_test': X_test_emb},
}
stacking_accuracies = {}
log_message("\n\n--- Champion Pipeline: Stacking with Various Meta-Learners ---")

# Base meta-features are the same for all configs
base_meta_features_train = [meta_features_train['MNB_tfidf'], meta_features_train['kNN_emb'], meta_features_train['DT_tfidf']]
base_meta_features_test = [test_predictions['MNB_tfidf'], test_predictions['kNN_emb'], test_predictions['DT_tfidf']]

for name, config in meta_learner_configs.items():
    meta_learner = config['learner']
    X_meta_train = config['X_train']
    X_meta_test = config['X_test']
    
    log_message("\n" + "="*50)
    log_message(f"Evaluating Meta-Learner: {name}")
    log_message("="*50)
    
    # Create the training set for the meta-learner
    is_sparse_train = any(issparse(arr) for arr in base_meta_features_train + [X_meta_train])
    if is_sparse_train:
        meta_learner_train_X = hstack(base_meta_features_train + [X_meta_train]).tocsr()
    else:
        meta_learner_train_X = np.hstack(base_meta_features_train + [X_meta_train])

    # Train the meta-learner
    print(f"  Training meta-learner {name}...")
    meta_learner.fit(meta_learner_train_X, y_train)
    
    # Create the test set for the meta-learner
    is_sparse_test = any(issparse(arr) for arr in base_meta_features_test + [X_meta_test])
    if is_sparse_test:
        meta_learner_test_X = hstack(base_meta_features_test + [X_meta_test]).tocsr()
    else:
        meta_learner_test_X = np.hstack(base_meta_features_test + [X_meta_test])

    # Make final predictions
    print(f"  Predicting with {name}...")
    final_preds = meta_learner.predict(meta_learner_test_X)
    
    # Calculate and store results
    accuracy = accuracy_score(y_test, final_preds)
    report = classification_report(y_test, final_preds, target_names=unique_labels, zero_division=0)
    stacking_accuracies[name] = accuracy
    
    log_message(f"Overall Accuracy: {accuracy:.4f}\n" + report)

# 7. Generate and Log Summary Table
summary_header = f"\n\n--- Champion Stacking Pipeline Summary (Accuracy) ---"
table_header = f"{'Meta-Learner Configuration':<35} | {'Accuracy':<15}"
separator = "-" * len(table_header)

log_message(summary_header, to_console=True)
log_message(table_header, to_console=True)
log_message(separator, to_console=True)

for name, accuracy in stacking_accuracies.items():
    row_str = f"{name:<35} | {accuracy:<15.4f}"
    log_message(row_str, to_console=True)

log_message(separator, to_console=True)
print(f"\nChampion pipeline benchmark complete. Results appended to '{LOG_FILE_PATH}'.")