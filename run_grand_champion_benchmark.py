# run_grand_champion_benchmark.py

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
from sklearn.metrics import accuracy_score, classification_report

# XGBoost for a powerful meta-learner
import xgboost as xgb

# Other utilities
from scipy.sparse import hstack, issparse
from tqdm.auto import tqdm
import torch

# --- Configuration ---
# NEW Data Scope
CATEGORIES_TO_SELECT = [
    'math', 'astro-ph', 'cs', 'cond-mat', 'physics', 
    'hep-ph', 'quant-ph', 'hep-th'
]
SAMPLES_PER_CATEGORY = 5000 # 40k total samples for a robust benchmark

# Models & Vectorizers
E5_MODEL_NAME = "intfloat/multilingual-e5-base"
TFIDF_MAX_FEATURES = 10000
RANDOM_STATE = 42
CV_FOLDS = 5 # Using 5 folds for more robust tuning and meta-feature generation
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_FILE_PATH = "grand_benchmark_results.txt"

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
domain_specific_stopwords = {
    'result', 'study', 'show', 'paper', 'model', 'analysis', 'method', 
    'approach', 'propose', 'demonstrate', 'investigate', 'present', 
    'based', 'using', 'also', 'however', 'provide', 'describe'
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

# --- Main Execution ---
log_message("\n\n" + "="*80)
log_message(f"--- Grand Champion Benchmark Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
log_message(f"--- Dataset: {len(CATEGORIES_TO_SELECT)} categories, {SAMPLES_PER_CATEGORY} samples/cat ---")
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
tfidf_vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, min_df=5, max_df=0.7, sublinear_tf=True, ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
X_test_tfidf = tfidf_vectorizer.transform(test_texts)
# Bag of Words (sharing vocab with TF-IDF)
print("Creating Bag of Words features...")
bow_vectorizer = CountVectorizer(vocabulary=tfidf_vectorizer.vocabulary_)
X_train_bow = bow_vectorizer.fit_transform(train_texts)
X_test_bow = bow_vectorizer.transform(test_texts)
# SBERT e5-base Embeddings
print(f"Creating SBERT Embeddings using {E5_MODEL_NAME}...")
sbert_model = SentenceTransformer(E5_MODEL_NAME, device=DEVICE)
X_train_emb = sbert_model.encode(train_texts, batch_size=BATCH_SIZE, show_progress_bar=True)
X_test_emb = sbert_model.encode(test_texts, batch_size=BATCH_SIZE, show_progress_bar=True)
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
dt_params = {'max_depth': [20, 30, 40], 'min_samples_leaf': [1, 5, 10]}
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_STATE), dt_params, cv=CV_FOLDS, n_jobs=-1)
dt_grid.fit(X_train_tfidf, y_train)
best_dt = dt_grid.best_estimator_
log_message(f"Best DT params: {dt_grid.best_params_}", to_console=True)

# 4. Evaluate Single Model Baselines
print("\n--- Step 4: Evaluating Single Model Baselines ---")
log_message("\n\n--- Grand Champion: Single Model Baselines ---")
results = {}
# MNB
mnb_preds = best_mnb.predict(X_test_tfidf)
results['MNB(tfidf)'] = accuracy_score(y_test, mnb_preds)
report = classification_report(y_test, mnb_preds, target_names=unique_labels, zero_division=0)
log_message("\n" + "="*50 + "\nModel: Tuned MNB(tfidf)\n" + "="*50 + f"\nOverall Accuracy: {results['MNB(tfidf)']:.4f}\n" + report)
# kNN
knn_preds = best_knn.predict(X_test_emb)
results['kNN(emb)'] = accuracy_score(y_test, knn_preds)
report = classification_report(y_test, knn_preds, target_names=unique_labels, zero_division=0)
log_message("\n" + "="*50 + "\nModel: Tuned kNN(emb)\n" + "="*50 + f"\nOverall Accuracy: {results['kNN(emb)']:.4f}\n" + report)
# LR
lr_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000).fit(X_train_tfidf, y_train)
lr_preds = lr_model.predict(X_test_tfidf)
results['LR(tfidf)'] = accuracy_score(y_test, lr_preds)
report = classification_report(y_test, lr_preds, target_names=unique_labels, zero_division=0)
log_message("\n" + "="*50 + "\nModel: LR(tfidf)\n" + "="*50 + f"\nOverall Accuracy: {results['LR(tfidf)']:.4f}\n" + report)

# 5. Generate Out-of-Fold Predictions for Stacking
print("\n--- Step 5: Calibrating and Generating Meta-Features ---")
calibrated_dt = CalibratedClassifierCV(estimator=best_dt, cv=CV_FOLDS, method='isotonic')
calibrated_knn = CalibratedClassifierCV(estimator=best_knn, cv=CV_FOLDS, method='isotonic')
base_models_for_stacking = {'MNB_tfidf': best_mnb, 'kNN_emb': calibrated_knn, 'DT_tfidf': calibrated_dt}
meta_features_train = {}
for name, model in base_models_for_stacking.items():
    feature_type = name.split('_')[1]
    X_for_model = X_train_tfidf if feature_type == 'tfidf' else X_train_emb
    meta_features_train[name] = cross_val_predict(model, X_for_model, y_train, cv=CV_FOLDS, method='predict_proba', n_jobs=-1)
calibrated_dt.fit(X_train_tfidf, y_train)
calibrated_knn.fit(X_train_emb, y_train)
print("All meta-features for stacking generated.")

# 6. Evaluate Soft Voting Ensemble
print("\n--- Step 6: Evaluating Soft Voting Ensemble ---")
mnb_probs = best_mnb.predict_proba(X_test_tfidf)
knn_probs = calibrated_knn.predict_proba(X_test_emb)
dt_probs = calibrated_dt.predict_proba(X_test_tfidf)
final_probs = (0.4 * mnb_probs) + (0.4 * knn_probs) + (0.2 * dt_probs)
soft_vote_preds = np.argmax(final_probs, axis=1)
results['Soft Voting'] = accuracy_score(y_test, soft_vote_preds)
report = classification_report(y_test, soft_vote_preds, target_names=unique_labels, zero_division=0)
log_message("\n" + "="*50 + "\nModel: Soft Voting Ensemble\n" + "="*50 + f"\nOverall Accuracy: {results['Soft Voting']:.4f}\n" + report)

# 7. Evaluate Stacking Ensembles
print("\n--- Step 7: Benchmarking Stacking Meta-Learner Configurations ---")
meta_learner_configs = {
    "Stack: LR(TFIDF)":  {'learner': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000), 'X_train': X_train_tfidf, 'X_test': X_test_tfidf},
    "Stack: LR(BoW)":    {'learner': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000), 'X_train': X_train_bow, 'X_test': X_test_bow},
    "Stack: LR(Emb)":    {'learner': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000), 'X_train': X_train_emb, 'X_test': X_test_emb},
    "Stack: XGB(TFIDF)": {'learner': xgb.XGBClassifier(random_state=RANDOM_STATE), 'X_train': X_train_tfidf, 'X_test': X_test_tfidf},
    "Stack: XGB(BoW)":   {'learner': xgb.XGBClassifier(random_state=RANDOM_STATE), 'X_train': X_train_bow, 'X_test': X_test_bow},
    "Stack: XGB(Emb)":   {'learner': xgb.XGBClassifier(random_state=RANDOM_STATE), 'X_train': X_train_emb, 'X_test': X_test_emb},
    "Stack: GNB(Emb)":   {'learner': GaussianNB(), 'X_train': X_train_emb, 'X_test': X_test_emb},
}
base_meta_features_train = [meta_features_train['MNB_tfidf'], meta_features_train['kNN_emb'], meta_features_train['DT_tfidf']]
base_meta_features_test = [mnb_probs, knn_probs, dt_probs]
for name, config in meta_learner_configs.items():
    meta_learner = config['learner']
    X_meta_train, X_meta_test = config['X_train'], config['X_test']
    print(f"  Training meta-learner {name}...")
    is_sparse = issparse(X_meta_train)
    meta_learner_train_X = hstack(base_meta_features_train + [X_meta_train]).tocsr() if is_sparse else np.hstack(base_meta_features_train + [X_meta_train])
    meta_learner.fit(meta_learner_train_X, y_train)
    meta_learner_test_X = hstack(base_meta_features_test + [X_meta_test]).tocsr() if is_sparse else np.hstack(base_meta_features_test + [X_meta_test])
    final_preds = meta_learner.predict(meta_learner_test_X)
    results[name] = accuracy_score(y_test, final_preds)
    report = classification_report(y_test, final_preds, target_names=unique_labels, zero_division=0)
    log_message("\n" + "="*50 + f"\nModel: {name}\n" + "="*50 + f"\nOverall Accuracy: {results[name]:.4f}\n" + report)

# 8. Generate Final Summary Table
summary_header = f"\n\n--- Grand Champion Benchmark Summary ({SAMPLES_PER_CATEGORY} samples/cat) ---"
table_header = f"{'Model Configuration':<35} | {'Accuracy':<15}"
separator = "-" * len(table_header)
log_message(summary_header, to_console=True)
log_message(table_header, to_console=True)
log_message(separator, to_console=True)
for name, accuracy in results.items():
    row_str = f"{name:<35} | {accuracy:<15.4f}"
    log_message(row_str, to_console=True)
log_message(separator, to_console=True)
print(f"\nGrand Champion benchmark complete. Results appended to '{LOG_FILE_PATH}'.")