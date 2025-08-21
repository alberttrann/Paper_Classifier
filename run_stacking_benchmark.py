# run_stacking_benchmark.py

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
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier # Used for cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Import issparse to check for sparse matrices
from scipy.sparse import hstack, issparse

from tqdm.auto import tqdm
import torch

# --- Configuration ---
CATEGORIES_TO_SELECT = ['astro-ph', 'cond-mat', 'cs', 'math', 'physics']
SAMPLES_PER_CATEGORY = 1000 # Using 1000 for a robust result (5k total)
DATASET_NAME = "UniverseTBD/arxiv-abstracts-large"
MODEL_NAME = "intfloat/multilingual-e5-base"
TFIDF_MAX_FEATURES = 10000
KNN_N_NEIGHBORS = 5
DT_MAX_DEPTH = 20
RANDOM_STATE = 42
CV_FOLDS = 5
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_FILE_PATH = "benchmark_results.txt"

# --- NLTK Downloads ---
# (Assuming they are already downloaded from previous runs)

# --- Helper function for logging ---
def log_message(message):
    print(message)
    with open(LOG_FILE_PATH, 'a') as f:
        f.write(message + '\n')

# --- Text Preprocessing Function ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(cleaned_tokens)

# --- Main Execution ---

log_message("\n\n" + "="*80)
log_message(f"--- Stacking Ensemble Benchmark Run (Extended): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
log_message("="*80)

# 1. Data Sampling and Preprocessing
print("--- Step 1: Data Sampling & Preprocessing ---")
# ... (This section is identical to previous scripts) ...
category_counts = {cat: 0 for cat in CATEGORIES_TO_SELECT}
samples = []
dataset_generator = load_dataset(DATASET_NAME, split="train", streaming=True)
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
processed_abstracts = [clean_text(abstract) for abstract in tqdm(abstracts, desc="Cleaning Text")]
unique_labels = sorted(list(set(labels_str)))
label_to_int = {label: i for i, label in enumerate(unique_labels)}
labels = np.array([label_to_int[label] for label in labels_str])
X_train_text, X_test_text, y_train, y_test = train_test_split(
    processed_abstracts, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
)

# 2. Feature Engineering
print("\n--- Step 2: Feature Engineering (All types) ---")
bow_vectorizer = CountVectorizer(max_features=TFIDF_MAX_FEATURES)
X_train_bow = bow_vectorizer.fit_transform(X_train_text)
X_test_bow = bow_vectorizer.transform(X_test_text)
tfidf_vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
sbert_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
X_train_emb = sbert_model.encode(X_train_text, batch_size=BATCH_SIZE, show_progress_bar=True)
X_test_emb = sbert_model.encode(X_test_text, batch_size=BATCH_SIZE, show_progress_bar=True)
print("All feature sets created.")

# 3. Generate Out-of-Fold Predictions (Meta-Features) for Training
print("\n--- Step 3: Generating Out-of-Fold Predictions for Meta-Learner Training ---")
meta_features_train = {}
base_models = {}
# MNB Meta-Features
print("Generating meta-features for MNB (BoW & TF-IDF)...")
base_models['MNB_bow'] = MultinomialNB()
meta_features_train['MNB_bow'] = cross_val_predict(base_models['MNB_bow'], X_train_bow, y_train, cv=CV_FOLDS, method='predict_proba')
base_models['MNB_tfidf'] = MultinomialNB()
meta_features_train['MNB_tfidf'] = cross_val_predict(base_models['MNB_tfidf'], X_train_tfidf, y_train, cv=CV_FOLDS, method='predict_proba')
# kNN Meta-Features
print("Generating meta-features for kNN (Embeddings)...")
base_models['kNN_emb'] = KNeighborsClassifier(n_neighbors=KNN_N_NEIGHBORS)
meta_features_train['kNN_emb'] = cross_val_predict(base_models['kNN_emb'], X_train_emb, y_train, cv=CV_FOLDS, method='predict_proba')
# DT Meta-Features
print("Generating meta-features for DT (BoW & TF-IDF)...")
base_models['DT_bow'] = DecisionTreeClassifier(max_depth=DT_MAX_DEPTH, random_state=RANDOM_STATE)
meta_features_train['DT_bow'] = cross_val_predict(base_models['DT_bow'], X_train_bow, y_train, cv=CV_FOLDS, method='predict_proba')
base_models['DT_tfidf'] = DecisionTreeClassifier(max_depth=DT_MAX_DEPTH, random_state=RANDOM_STATE)
meta_features_train['DT_tfidf'] = cross_val_predict(base_models['DT_tfidf'], X_train_tfidf, y_train, cv=CV_FOLDS, method='predict_proba')
print("All meta-features for training generated.")

# 4. Train Base Models on Full Training Data
print("\n--- Step 4: Training Base Models on Full Training Data ---")
for name, model in base_models.items():
    feature_type = name.split('_')[1]
    if feature_type == 'bow':
        model.fit(X_train_bow, y_train)
    elif feature_type == 'tfidf':
        model.fit(X_train_tfidf, y_train)
    else: # emb
        model.fit(X_train_emb, y_train)
print("Base models trained.")

# 5. Define Stacking Configurations and Benchmark
print("\n--- Step 5: Benchmarking Stacking Configurations ---")
stacking_configs = {
    "Stack 1: [MNB(b)+kNN(e)+DT(t)] + LR(b)":  {'base': ['MNB_bow', 'kNN_emb', 'DT_tfidf'], 'meta_learner': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000), 'meta_features': X_train_bow, 'test_features': X_test_bow},
    "Stack 2: [MNB(b)+kNN(e)+DT(t)] + LR(t)":  {'base': ['MNB_bow', 'kNN_emb', 'DT_tfidf'], 'meta_learner': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000), 'meta_features': X_train_tfidf, 'test_features': X_test_tfidf},
    "Stack 3: [MNB(b)+kNN(e)+DT(t)] + LR(e)":  {'base': ['MNB_bow', 'kNN_emb', 'DT_tfidf'], 'meta_learner': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000), 'meta_features': X_train_emb, 'test_features': X_test_emb},
    "Stack 4: [MNB(t)+kNN(e)+DT(t)] + LR(b)":  {'base': ['MNB_tfidf', 'kNN_emb', 'DT_tfidf'], 'meta_learner': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000), 'meta_features': X_train_bow, 'test_features': X_test_bow},
    "Stack 5: [MNB(t)+kNN(e)+DT(t)] + LR(t)":  {'base': ['MNB_tfidf', 'kNN_emb', 'DT_tfidf'], 'meta_learner': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000), 'meta_features': X_train_tfidf, 'test_features': X_test_tfidf},
    "Stack 6: [MNB(t)+kNN(e)+DT(t)] + LR(e)":  {'base': ['MNB_tfidf', 'kNN_emb', 'DT_tfidf'], 'meta_learner': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000), 'meta_features': X_train_emb, 'test_features': X_test_emb},
    "Stack 7: [MNB(b)+kNN(e)+DT(t)] + DT(t)":  {'base': ['MNB_bow', 'kNN_emb', 'DT_tfidf'], 'meta_learner': DecisionTreeClassifier(max_depth=DT_MAX_DEPTH, random_state=RANDOM_STATE), 'meta_features': X_train_tfidf, 'test_features': X_test_tfidf},
    # NEW CONFIGURATIONS
    "Stack 8: [MNB(b)+kNN(e)] + DT(t)":        {'base': ['MNB_bow', 'kNN_emb'], 'meta_learner': DecisionTreeClassifier(max_depth=DT_MAX_DEPTH, random_state=RANDOM_STATE), 'meta_features': X_train_tfidf, 'test_features': X_test_tfidf},
    "Stack 9: [MNB(b)+kNN(e)] + LR(b)":        {'base': ['MNB_bow', 'kNN_emb'], 'meta_learner': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000), 'meta_features': X_train_bow, 'test_features': X_test_bow},
    "Stack 10: [MNB(b)+kNN(e)] + LR(t)":       {'base': ['MNB_bow', 'kNN_emb'], 'meta_learner': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000), 'meta_features': X_train_tfidf, 'test_features': X_test_tfidf},
    "Stack 11: [MNB(b)+kNN(e)] + LR(e)":       {'base': ['MNB_bow', 'kNN_emb'], 'meta_learner': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000), 'meta_features': X_train_emb, 'test_features': X_test_emb}
}
stacking_accuracies = {}

log_message("\n\n--- Detailed Stacking Ensemble Reports (Extended) ---")

for name, config in stacking_configs.items():
    base_components = config['base']
    meta_learner = config['meta_learner']
    X_meta_train = config['meta_features']
    X_meta_test = config['test_features']
    
    component_str = " + ".join([c.replace('_', '(') + ')' for c in base_components])
    log_message("\n" + "="*50)
    log_message(f"Evaluating: {name} | Base: {component_str}")
    log_message("="*50)
    
    # Create the training set for the meta-learner
    train_meta_list = [meta_features_train[comp] for comp in base_components]
    
    # Check if any of the components to be stacked are sparse matrices
    is_sparse_train = any(issparse(arr) for arr in train_meta_list + [X_meta_train])
    if is_sparse_train:
        meta_learner_train_X = hstack(train_meta_list + [X_meta_train]).tocsr()
    else: # All inputs are dense, use numpy's hstack
        meta_learner_train_X = np.hstack(train_meta_list + [X_meta_train])

    # Train the meta-learner
    print(f"  Training meta-learner for {name}...")
    meta_learner.fit(meta_learner_train_X, y_train)
    
    # Create the test set for the meta-learner
    test_pred_list = [base_models[comp].predict_proba(X_test_bow if 'bow' in comp else (X_test_tfidf if 'tfidf' in comp else X_test_emb)) for comp in base_components]
    
    is_sparse_test = any(issparse(arr) for arr in test_pred_list + [X_meta_test])
    if is_sparse_test:
        meta_learner_test_X = hstack(test_pred_list + [X_meta_test]).tocsr()
    else:
        meta_learner_test_X = np.hstack(test_pred_list + [X_meta_test])

    # Make final predictions
    print(f"  Predicting with {name}...")
    final_preds = meta_learner.predict(meta_learner_test_X)
    
    # Calculate and store results
    accuracy = accuracy_score(y_test, final_preds)
    report = classification_report(y_test, final_preds, target_names=unique_labels, zero_division=0)
    
    stacking_accuracies[name] = accuracy
    
    log_message(f"Overall Accuracy: {accuracy:.4f}\n")
    log_message(report)

# 6. Generate and Log Summary Table
summary_header = f"\n\n--- Stacking Ensemble Summary (Accuracy) ---"
table_header = f"{'Stacking Configuration':<65} | {'Accuracy':<15}"
separator = "-" * len(table_header)

log_message(summary_header)
log_message(table_header)
log_message(separator)
print(summary_header)
print(table_header)
print(separator)

for name, config in stacking_configs.items():
    base_str = " + ".join([c.replace('_', '(') + ')' for c in config['base']])
    meta_learner_info = str(config['meta_learner']).split('(')[0]
    
    # Determine feature type for meta-learner for display
    if issparse(config['meta_features']):
        if 'Tfidf' in config['meta_features'].__class__.__name__:
            meta_feature_str = 'TFIDF'
        else:
            meta_feature_str = 'BoW'
    else:
        meta_feature_str = 'Emb'
        
    meta_str = f"{meta_learner_info}({meta_feature_str})"
    full_config_str = f"[{base_str}] + {meta_str}"
    
    accuracy = stacking_accuracies.get(name, 0) # Use .get for safety
    row_str = f"{full_config_str:<65} | {accuracy:<15.4f}"
    log_message(row_str)
    print(row_str)

log_message(separator)
print(separator)

print(f"\nStacking benchmark complete. All results have been appended to '{LOG_FILE_PATH}'.")