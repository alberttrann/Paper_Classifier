# run_multilabel_grand_prix.py

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

# Sentence Transformers for embeddings
from sentence_transformers import SentenceTransformer

# Scikit-learn for models, vectorizers, and metrics
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, hamming_loss

# XGBoost
import xgboost as xgb

# Other utilities
from scipy.sparse import hstack, issparse, csr_matrix
from tqdm.auto import tqdm
import torch

# --- Configuration ---
CATEGORIES_TO_SELECT = ['math', 'astro-ph', 'cs', 'cond-mat', 'physics', 'hep-ph', 'quant-ph', 'hep-th']
SAMPLES_PER_CATEGORY_APPEARANCE = 5000 # Reduced slightly to manage extreme runtime

E5_MODEL_NAME = "intfloat/multilingual-e5-base"
TFIDF_MAX_FEATURES = 10000
RANDOM_STATE = 42
CV_FOLDS = 5 # Use 5 folds for faster tuning and stacking
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_FILE_PATH = "multilabel_grand_prix.txt"

# --- Helper function for logging ---
def log_message(message, to_console=True):
    if to_console:
        print(message)
    with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

# --- Enhanced Text Preprocessing Function ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
domain_specific_stopwords = {'result', 'study', 'show', 'paper', 'model', 'analysis', 'method', 'approach', 'propose', 'demonstrate', 'investigate'}
stop_words.update(domain_specific_stopwords)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(cleaned_tokens)

# --- K-Means for Classification Function ---
def train_and_predict_kmeans(X_train, y_train, X_test):
    kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)
    cluster_ids_train = kmeans.fit_predict(X_train)
    
    # Assign a class label to each cluster (0 or 1)
    cluster_to_label = {}
    for cluster_id in [0, 1]:
        labels_in_cluster = y_train[cluster_ids_train == cluster_id]
        if len(labels_in_cluster) == 0:
            most_common_label = Counter(y_train).most_common(1)[0][0]
        else:
            most_common_label = Counter(labels_in_cluster).most_common(1)[0][0]
        cluster_to_label[cluster_id] = most_common_label
        
    cluster_ids_test = kmeans.predict(X_test)
    predictions = np.array([cluster_to_label.get(cid, 0) for cid in cluster_ids_test])
    return predictions

# --- Main Execution ---
log_message("\n\n" + "="*80)
log_message(f"--- Multi-Label Grand Prix Benchmark: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
log_message("="*80)

# 1. Multi-Label Data Sampling and Preparation
print("--- Step 1: Multi-Label Data Sampling & Preparation ---")
# ... (Same as before) ...
category_counts = {cat: 0 for cat in CATEGORIES_TO_SELECT}
samples = []
dataset_generator = load_dataset("UniverseTBD/arxiv-abstracts-large", split="train", streaming=True)
for s in tqdm(dataset_generator, desc="Scanning for samples"):
    if all(count >= SAMPLES_PER_CATEGORY_APPEARANCE for count in category_counts.values()):
        break
    if s['categories'] is None or s['abstract'] is None: continue
    parent_categories = {cat.split('.')[0] for cat in s['categories'].strip().split(' ')}
    if any(p in CATEGORIES_TO_SELECT for p in parent_categories):
        samples.append(s)
        for p_cat in parent_categories:
            if p_cat in category_counts:
                category_counts[p_cat] += 1
print(f"Finished sampling. Total samples collected: {len(samples)}")
abstracts = [sample['abstract'] for sample in samples]
labels_sets = [{cat.split('.')[0] for cat in sample['categories'].strip().split(' ')} for sample in samples]
processed_abstracts = [clean_text(abstract) for abstract in tqdm(abstracts, desc="Cleaning Abstracts")]
Y = np.zeros((len(samples), len(CATEGORIES_TO_SELECT)), dtype=int)
cat_to_idx = {cat: i for i, cat in enumerate(CATEGORIES_TO_SELECT)}
for i, label_set in enumerate(labels_sets):
    for label in label_set:
        if label in cat_to_idx:
            Y[i, cat_to_idx[label]] = 1
train_texts, test_texts, Y_train, Y_test = train_test_split(processed_abstracts, Y, test_size=0.2, random_state=RANDOM_STATE)

# 2. Feature Engineering
print("\n--- Step 2: Enhanced Feature Engineering ---")
# ... (Same as before) ...
tfidf_vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, min_df=5, max_df=0.7, sublinear_tf=True, ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
X_test_tfidf = tfidf_vectorizer.transform(test_texts)
bow_vectorizer = CountVectorizer(vocabulary=tfidf_vectorizer.vocabulary_)
X_train_bow = bow_vectorizer.fit_transform(train_texts)
X_test_bow = bow_vectorizer.transform(test_texts)
sbert_model = SentenceTransformer(E5_MODEL_NAME, device=DEVICE)
X_train_emb = sbert_model.encode(train_texts, batch_size=BATCH_SIZE, show_progress_bar=True)
X_test_emb = sbert_model.encode(test_texts, batch_size=BATCH_SIZE, show_progress_bar=True)
print("All feature sets created.")

# 3. Define Benchmark Architectures
architectures = {
    # Singles
    "LR(TFIDF)": {"type": "single", "model": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced'), "X_train": X_train_tfidf, "X_test": X_test_tfidf},
    "LR(BoW)": {"type": "single", "model": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced'), "X_train": X_train_bow, "X_test": X_test_bow},
    "LR(Emb)": {"type": "single", "model": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced'), "X_train": X_train_emb, "X_test": X_test_emb},
    "XGB(TFIDF)": {"type": "single", "model": xgb.XGBClassifier(random_state=RANDOM_STATE), "X_train": X_train_tfidf, "X_test": X_test_tfidf},
    "XGB(BoW)": {"type": "single", "model": xgb.XGBClassifier(random_state=RANDOM_STATE), "X_train": X_train_bow, "X_test": X_test_bow},
    "XGB(Emb)": {"type": "single", "model": xgb.XGBClassifier(random_state=RANDOM_STATE), "X_train": X_train_emb, "X_test": X_test_emb},
    "kNN(Emb)": {"type": "single", "model": KNeighborsClassifier(n_neighbors=7, weights='distance'), "X_train": X_train_emb, "X_test": X_test_emb},
    "GNB(TFIDF)": {"type": "single", "model": GaussianNB(), "X_train": X_train_tfidf.toarray(), "X_test": X_test_tfidf.toarray()},
    "GNB(BoW)": {"type": "single", "model": GaussianNB(), "X_train": X_train_bow.toarray(), "X_test": X_test_bow.toarray()},
    "GNB(Emb)": {"type": "single", "model": GaussianNB(), "X_train": X_train_emb, "X_test": X_test_emb},
    "KMeans(Emb)": {"type": "kmeans", "X_train": X_train_emb, "X_test": X_test_emb},
    # Voting Ensembles
    "VoteEns_1": {"type": "vote", "components": ["MNB(bow)", "kNN(emb)", "DT(tfidf)"]},
    "VoteEns_2": {"type": "vote", "components": ["MNB(tfidf)", "kNN(emb)", "DT(tfidf)"]},
    # Stacking Ensembles
    "Stack_LR(t)": {"type": "stack", "meta_learner": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced'), "meta_X_train": X_train_tfidf, "meta_X_test": X_test_tfidf},
    "Pure_Stack_LR": {"type": "stack", "meta_learner": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced'), "meta_X_train": None, "meta_X_test": None}
}
results = {}

# 4. Run the Grand Prix
log_message("\n\n--- Detailed Grand Prix Reports ---")
for arch_name, config in architectures.items():
    print(f"\n--- Benchmarking Architecture: {arch_name} ---")
    log_message("\n" + "="*50 + f"\nArchitecture: {arch_name}\n" + "="*50)
    Y_pred = np.zeros_like(Y_test)

    for i, category in enumerate(CATEGORIES_TO_SELECT):
        print(f"  Training for category: {category}")
        y_binary_train = Y_train[:, i]
        y_binary_test = Y_test[:, i]
        
        # --- PREDICTION LOGIC ---
        if config["type"] == "single":
            model = config["model"]
            model.fit(config["X_train"], y_binary_train)
            Y_pred[:, i] = model.predict(config["X_test"])
        
        elif config["type"] == "kmeans":
            Y_pred[:, i] = train_and_predict_kmeans(config["X_train"], y_binary_train, config["X_test"])
        
        else: # Ensembles (Voting or Stacking)
            # Define and tune base models for this binary task
            mnb_bow = MultinomialNB(alpha=0.1).fit(X_train_bow, y_binary_train)
            mnb_tfidf = MultinomialNB(alpha=0.1).fit(X_train_tfidf, y_binary_train)
            knn_emb = KNeighborsClassifier(n_neighbors=7, weights='distance').fit(X_train_emb, y_binary_train)
            dt_tfidf = DecisionTreeClassifier(max_depth=40, random_state=RANDOM_STATE, class_weight='balanced').fit(X_train_tfidf, y_binary_train)
            
            calibrated_knn = CalibratedClassifierCV(estimator=knn_emb, cv=CV_FOLDS).fit(X_train_emb, y_binary_train)
            calibrated_dt = CalibratedClassifierCV(estimator=dt_tfidf, cv=CV_FOLDS).fit(X_train_tfidf, y_binary_train)

            if config["type"] == "vote":
                base_preds = {
                    "MNB(bow)": mnb_bow.predict_proba(X_test_bow)[:, 1],
                    "MNB(tfidf)": mnb_tfidf.predict_proba(X_test_tfidf)[:, 1],
                    "kNN(emb)": calibrated_knn.predict_proba(X_test_emb)[:, 1],
                    "DT(tfidf)": calibrated_dt.predict_proba(X_test_tfidf)[:, 1]
                }
                # Weighted average of probabilities
                probs = np.mean([base_preds[comp] for comp in config["components"]], axis=0)
                Y_pred[:, i] = (probs >= 0.5).astype(int)

            elif config["type"] == "stack":
                # Stacking logic
                base_models_for_stacking = {'MNB_tfidf': mnb_tfidf, 'kNN_emb': knn_emb, 'DT_tfidf': dt_tfidf}
                meta_features_train = []
                meta_features_test = []
                
                for name, model in base_models_for_stacking.items():
                    feature_type = name.split('_')[1]
                    X_for_model_train = X_train_tfidf if feature_type == 'tfidf' else X_train_emb
                    X_for_model_test = X_test_tfidf if feature_type == 'tfidf' else X_test_emb
                    
                    meta_features_train.append(cross_val_predict(model, X_for_model_train, y_binary_train, cv=CV_FOLDS, method='predict_proba', n_jobs=-1)[:, 1])
                    meta_features_test.append(model.predict_proba(X_for_model_test)[:, 1])
                
                meta_features_train = np.stack(meta_features_train, axis=1)
                meta_features_test = np.stack(meta_features_test, axis=1)

                if config["meta_X_train"] is not None:
                    is_sparse = issparse(config["meta_X_train"])
                    meta_learner_train_X = hstack([csr_matrix(meta_features_train), config["meta_X_train"]]).tocsr() if is_sparse else np.hstack([meta_features_train, config["meta_X_train"]])
                    meta_learner_test_X = hstack([csr_matrix(meta_features_test), config["meta_X_test"]]).tocsr() if is_sparse else np.hstack([meta_features_test, config["meta_X_test"]])
                else: # Pure stack
                    meta_learner_train_X = meta_features_train
                    meta_learner_test_X = meta_features_test
                
                meta_learner = config["meta_learner"]
                meta_learner.fit(meta_learner_train_X, y_binary_train)
                Y_pred[:, i] = meta_learner.predict(meta_learner_test_X)

    # Evaluate the architecture
    acc = accuracy_score(Y_test, Y_pred)
    ham = hamming_loss(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred, target_names=CATEGORIES_TO_SELECT, zero_division=0)
    
    results[arch_name] = {'accuracy': acc, 'hamming': ham}
    log_message(f"Overall Subset Accuracy: {acc:.4f}")
    log_message(f"Hamming Loss: {ham:.4f}\n" + report)

# 5. Final Summary Table
summary_header = f"\n\n--- Multi-Label Grand Prix Summary ---"
table_header = f"{'Model Architecture':<45} | {'Subset Accuracy':<20} | {'Hamming Loss (Lower is Better)':<30}"
separator = "-" * len(table_header)
log_message(summary_header, to_console=True)
log_message(table_header, to_console=True)
log_message(separator, to_console=True)
# Sort results by subset accuracy for a ranked list
sorted_results = sorted(results.items(), key=lambda item: item[1]['accuracy'], reverse=True)
for name, metrics in sorted_results:
    row_str = f"{name:<45} | {metrics['accuracy']:<20.4f} | {metrics['hamming']:<30.4f}"
    log_message(row_str, to_console=True)
log_message(separator, to_console=True)

print(f"\nMulti-label Grand Prix complete. Results appended to '{LOG_FILE_PATH}'.")