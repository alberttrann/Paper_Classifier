# run_ultimate_benchmark.py

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

# --- MODIFIED ---
# We now only need the base SentenceTransformer class for e5-base
from sentence_transformers import SentenceTransformer

# Scikit-learn for models, vectorizers, and metrics
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


# Other utilities
from scipy.sparse import hstack, csr_matrix, issparse
from scipy.spatial.distance import jensenshannon
from tqdm.auto import tqdm
import torch

# --- Configuration ---
# Data Sampling
CATEGORIES_TO_SELECT = ['astro-ph', 'cond-mat', 'cs', 'math', 'physics']
SAMPLES_PER_CATEGORY = 2000 # Using 2000 for a robust result (10k total)

# --- MODIFIED ---
# Models & Vectorizers
E5_MODEL_NAME = "intfloat/multilingual-e5-base" # Swapped back to e5-base
TFIDF_MAX_FEATURES = 10000
KNN_N_NEIGHBORS = 5
DT_MAX_DEPTH = 20
RANDOM_STATE = 42
CV_FOLDS = 5 # 5 folds for faster GridSearchCV
BATCH_SIZE = 128 # e5 is generally efficient, can use a slightly larger batch size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_FILE_PATH = "benchmark_results.txt"

# --- NLTK Downloads ---
# (Assuming they are already downloaded from previous runs)

# --- Helper function for logging ---
def log_message(message, to_console=True):
    if to_console:
        print(message)
    with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

# --- Enhanced Text Preprocessing Function (with Custom Stop Words) ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
# Add your custom, domain-specific stop words
domain_specific_stopwords = {'result', 'study', 'show', 'paper', 'model', 'analysis', 'method', 'approach', 'propose', 'demonstrate', 'investigate', 'present', 'based', 'using', 'also', 'however', 'propose', 'provide', 'describe'}
stop_words.update(domain_specific_stopwords)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(cleaned_tokens)

# --- Main Execution ---
log_message("\n\n" + "="*80)
# --- MODIFIED ---
log_message(f"--- Ultimate Benchmark Run (e5-base Model): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
log_message("="*80)

# 1. Data Sampling and Preprocessing
print("--- Step 1: Data Sampling & Preprocessing ---")
category_counts = {cat: 0 for cat in CATEGORIES_TO_SELECT}
samples = []
dataset_generator = load_dataset("UniverseTBD/arxiv-abstracts-large", split="train", streaming=True)
for s in tqdm(dataset_generator, desc="Scanning for samples"):
    if all(count >= SAMPLES_PER_CATEGORY for count in category_counts.values()):
        break
    if s['categories'] is None or s['abstract'] is None or s['title'] is None or len(s['categories'].split(' ')) != 1:
        continue
    parent_category = s['categories'].strip().split('.')[0]
    if parent_category in CATEGORIES_TO_SELECT and category_counts[parent_category] < SAMPLES_PER_CATEGORY:
        s['parent_category'] = parent_category
        samples.append(s)
        category_counts[parent_category] += 1
print(f"Finished sampling. Total samples collected: {len(samples)}")
abstracts = [sample['abstract'] for sample in samples]
titles = [sample['title'] for sample in samples]
labels_str = [sample['parent_category'] for sample in samples]
processed_abstracts = [clean_text(abstract) for abstract in tqdm(abstracts, desc="Cleaning Abstracts")]
processed_titles = [clean_text(title) for title in tqdm(titles, desc="Cleaning Titles")]
unique_labels = sorted(list(set(labels_str)))
label_to_int = {label: i for i, label in enumerate(unique_labels)}
labels = np.array([label_to_int[label] for label in labels_str])
# Split indices to easily partition all the different feature sets later
indices = np.arange(len(samples))
train_indices, test_indices, y_train, y_test = train_test_split(
    indices, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
)

# 2. Feature Engineering (Phase 1: Base Features)
print("\n--- Step 2.1: Engineering Base Features ---")
# Advanced TF-IDF
print("Creating Advanced TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES, min_df=5, max_df=0.7, 
    sublinear_tf=True, ngram_range=(1, 2)
)
X_train_tfidf = tfidf_vectorizer.fit_transform([processed_abstracts[i] for i in train_indices])
X_test_tfidf = tfidf_vectorizer.transform([processed_abstracts[i] for i in test_indices])

# --- MODIFIED ---
# SBERT Embeddings (using e5-base)
print(f"Creating SBERT Embeddings using {E5_MODEL_NAME}...")
sbert_model = SentenceTransformer(E5_MODEL_NAME, device=DEVICE)
print("e5-base model created successfully.")

X_train_emb = sbert_model.encode([processed_abstracts[i] for i in train_indices], batch_size=BATCH_SIZE, show_progress_bar=True)
X_test_emb = sbert_model.encode([processed_abstracts[i] for i in test_indices], batch_size=BATCH_SIZE, show_progress_bar=True)

# 3. Feature Engineering (Phase 2: Advanced & Weird Features)
print("\n--- Step 2.2: Engineering Advanced & Novel Features ---")
# Metadata Features
print("Engineering Metadata features...")
train_meta = np.array([[len(abstracts[i]), len(abstracts[i].split()), len(abstracts[i]) / (len(abstracts[i].split()) + 1), abstracts[i].count('$')] for i in train_indices])
test_meta = np.array([[len(abstracts[i]), len(abstracts[i].split()), len(abstracts[i]) / (len(abstracts[i].split()) + 1), abstracts[i].count('$')] for i in test_indices])

# Title vs. Abstract Features
print("Engineering Title vs. Abstract features...")
train_titles_emb = sbert_model.encode([processed_titles[i] for i in train_indices], batch_size=BATCH_SIZE, show_progress_bar=True)
test_titles_emb = sbert_model.encode([processed_titles[i] for i in test_indices], batch_size=BATCH_SIZE, show_progress_bar=True)
# L2 Normalize for cosine similarity calculation
train_abs_norm = X_train_emb / np.linalg.norm(X_train_emb, axis=1, keepdims=True)
train_titles_norm = train_titles_emb / np.linalg.norm(train_titles_emb, axis=1, keepdims=True)
test_abs_norm = X_test_emb / np.linalg.norm(X_test_emb, axis=1, keepdims=True)
test_titles_norm = test_titles_emb / np.linalg.norm(test_titles_emb, axis=1, keepdims=True)
X_train_title_sim = (train_abs_norm * train_titles_norm).sum(axis=1).reshape(-1, 1)
X_test_title_sim = (test_abs_norm * test_titles_norm).sum(axis=1).reshape(-1, 1)

# Combined Feature Sets
print("Creating combined feature sets...")
X_train_tfidf_plus_emb = hstack([X_train_tfidf, csr_matrix(X_train_emb)])
X_test_tfidf_plus_emb = hstack([X_test_tfidf, csr_matrix(X_test_emb)])

# 4. Hyperparameter Tuning of Base Models
print("\n--- Step 3: Hyperparameter Tuning of Base Models ---")
# MNB
print("Tuning MultinomialNB...")
mnb_params = {'alpha': [0.01, 0.1, 0.5, 1.0]}
mnb_grid = GridSearchCV(MultinomialNB(), mnb_params, cv=CV_FOLDS, n_jobs=-1)
mnb_grid.fit(X_train_tfidf, y_train)
best_mnb = mnb_grid.best_estimator_
print(f"Best MNB params: {mnb_grid.best_params_}")

# kNN
print("Tuning KNeighborsClassifier...")
knn_params = {'n_neighbors': [5, 7, 11], 'weights': ['uniform', 'distance']}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=CV_FOLDS, n_jobs=-1)
knn_grid.fit(X_train_emb, y_train)
best_knn = knn_grid.best_estimator_
print(f"Best kNN params: {knn_grid.best_params_}")

# DT
print("Tuning DecisionTreeClassifier...")
dt_params = {'max_depth': [15, 20, 30], 'min_samples_leaf': [1, 5, 10]}
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_STATE), dt_params, cv=CV_FOLDS, n_jobs=-1)
dt_grid.fit(X_train_tfidf, y_train)
best_dt = dt_grid.best_estimator_
print(f"Best DT params: {dt_grid.best_params_}")

# 5. Advanced Single Model Benchmarks
print("\n--- Step 4: Benchmarking Models on Combined Features ---")
log_message("\n\n--- Advanced Single Model Reports (e5-base Model) ---", to_console=False)
# Logistic Regression on TF-IDF + Embeddings
print("Training LR on TF-IDF + Embeddings...")
lr_combined = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000).fit(X_train_tfidf_plus_emb, y_train)
lr_preds = lr_combined.predict(X_test_tfidf_plus_emb)
acc = accuracy_score(y_test, lr_preds)
report = classification_report(y_test, lr_preds, target_names=unique_labels, zero_division=0)
log_message("\n" + "="*50 + "\nModel: LogisticRegression on TF-IDF + Embeddings\n" + "="*50)
log_message(f"Overall Accuracy: {acc:.4f}\n" + report, to_console=False)

# 6. Calibrate Models and Generate Out-of-Fold Predictions for Stacking
print("\n--- Step 5: Calibrating Models and Generating Meta-Features ---")
# Calibrate the best-tuned DT and kNN
# --- MODIFIED ---
# Use the 'estimator' parameter for newer scikit-learn versions
calibrated_dt = CalibratedClassifierCV(estimator=best_dt, cv=CV_FOLDS, method='isotonic')
calibrated_knn = CalibratedClassifierCV(estimator=best_knn, cv=CV_FOLDS, method='isotonic')
# MNB is usually well-calibrated, so we'll use the tuned version directly
base_models_for_stacking = {
    'MNB_tfidf': best_mnb,
    'kNN_emb': calibrated_knn,
    'DT_tfidf': calibrated_dt
}
meta_features_train = {}
for name, model in base_models_for_stacking.items():
    feature_type = name.split('_')[1]
    print(f"Generating out-of-fold predictions for {name}...")
    X_for_model = X_train_tfidf if feature_type == 'tfidf' else X_train_emb
    meta_features_train[name] = cross_val_predict(model, X_for_model, y_train, cv=CV_FOLDS, method='predict_proba', n_jobs=-1)
# Also train the calibrated models on the full training data for later use
calibrated_dt.fit(X_train_tfidf, y_train)
calibrated_knn.fit(X_train_emb, y_train)
print("All meta-features for stacking generated.")

# 7. Final Stacking and Ensemble Benchmarks
print("\n--- Step 6: Final Ensemble Benchmarks ---")
log_message("\n\n--- Final Ensemble Reports (e5-base Model) ---", to_console=False)

# Soft Voting Ensemble
print("Evaluating Soft Voting Ensemble...")
mnb_probs = best_mnb.predict_proba(X_test_tfidf)
knn_probs = calibrated_knn.predict_proba(X_test_emb)
dt_probs = calibrated_dt.predict_proba(X_test_tfidf)
# Weighted average of probabilities
final_probs = (0.4 * mnb_probs) + (0.4 * knn_probs) + (0.2 * dt_probs)
soft_vote_preds = np.argmax(final_probs, axis=1)
acc = accuracy_score(y_test, soft_vote_preds)
report = classification_report(y_test, soft_vote_preds, target_names=unique_labels, zero_division=0)
log_message("\n" + "="*50 + "\nModel: Soft Voting Ensemble [MNB(t)+kNN(e)+DT(t)]\n" + "="*50)
log_message(f"Overall Accuracy: {acc:.4f}\n" + report, to_console=False)

# "Pure" Stacking (Probabilities Only)
print("Evaluating 'Pure' Stacking Ensemble...")
meta_learner_pure = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
meta_features_pure_train = np.hstack([meta_features_train['MNB_tfidf'], meta_features_train['kNN_emb'], meta_features_train['DT_tfidf']])
meta_learner_pure.fit(meta_features_pure_train, y_train)
meta_features_pure_test = np.hstack([mnb_probs, knn_probs, dt_probs])
pure_stack_preds = meta_learner_pure.predict(meta_features_pure_test)
acc = accuracy_score(y_test, pure_stack_preds)
report = classification_report(y_test, pure_stack_preds, target_names=unique_labels, zero_division=0)
log_message("\n" + "="*50 + "\nModel: Pure Stacking [MNB(t)+kNN(e)+DT(t)] + LR\n" + "="*50)
log_message(f"Overall Accuracy: {acc:.4f}\n" + report, to_console=False)

# Stacking with GNB meta-learner and Metadata features
print("Evaluating Stacking with GNB and Metadata...")
meta_learner_gnb = GaussianNB()
meta_features_gnb_train = np.hstack([meta_features_pure_train, train_meta, X_train_title_sim])
meta_learner_gnb.fit(meta_features_gnb_train, y_train)
meta_features_gnb_test = np.hstack([meta_features_pure_test, test_meta, X_test_title_sim])
gnb_stack_preds = meta_learner_gnb.predict(meta_features_gnb_test)
acc = accuracy_score(y_test, gnb_stack_preds)
report = classification_report(y_test, gnb_stack_preds, target_names=unique_labels, zero_division=0)
log_message("\n" + "="*50 + "\nModel: Stacking [Base Models] + GNB(meta+title)\n" + "="*50)
log_message(f"Overall Accuracy: {acc:.4f}\n" + report, to_console=False)

# Confidence-Gated Ensemble
print("Evaluating Confidence-Gated Ensemble...")
gatekeeper_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
# Create binary target: was the out-of-fold MNB prediction correct?
y_is_mnb_correct = (np.argmax(meta_features_train['MNB_tfidf'], axis=1) == y_train).astype(int)
gatekeeper_model.fit(X_train_tfidf, y_is_mnb_correct)
# Inference
gatekeeper_probs_correct = gatekeeper_model.predict_proba(X_test_tfidf)[:, 1]
mnb_final_preds = best_mnb.predict(X_test_tfidf)
knn_final_preds = calibrated_knn.predict(X_test_emb)
confidence_preds = np.where(gatekeeper_probs_correct >= 0.8, mnb_final_preds, knn_final_preds)
acc = accuracy_score(y_test, confidence_preds)
report = classification_report(y_test, confidence_preds, target_names=unique_labels, zero_division=0)
log_message("\n" + "="*50 + "\nModel: Confidence-Gated Ensemble [MNB(t) -> kNN(e)]\n" + "="*50)
log_message(f"Overall Accuracy: {acc:.4f}\n" + report, to_console=False)

print(f"\nUltimate benchmark complete. All results have been appended to '{LOG_FILE_PATH}'.")