# run_multilabel_championship.py

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
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, hamming_loss

# Other utilities
from scipy.sparse import hstack, issparse, csr_matrix
from tqdm.auto import tqdm
import torch

# --- Configuration ---
# Data Sampling
CATEGORIES_TO_SELECT = [
    'math', 'astro-ph', 'cs', 'cond-mat', 'physics', 
    'hep-ph', 'quant-ph', 'hep-th'
]
# We'll use a slightly smaller sample size to make the multiple stack training feasible
SAMPLES_PER_CATEGORY_APPEARANCE = 5000 
# Note: The final dataset size will be larger than 8 * 5000 due to multi-label overlaps

# Models & Vectorizers
E5_MODEL_NAME = "intfloat/multilingual-e5-base"
TFIDF_MAX_FEATURES = 10000
RANDOM_STATE = 42
CV_FOLDS = 5 # Use 3 folds for faster GridSearchCV and stacking
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_FILE_PATH = "multilabel_benchmarks.txt"

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
log_message(f"--- Multi-Label Championship Benchmark: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
log_message("="*80)

# 1. Multi-Label Data Sampling and Preparation
print("--- Step 1: Multi-Label Data Sampling & Preparation ---")
category_counts = {cat: 0 for cat in CATEGORIES_TO_SELECT}
samples = []
dataset_generator = load_dataset("UniverseTBD/arxiv-abstracts-large", split="train", streaming=True)
for s in tqdm(dataset_generator, desc="Scanning for samples"):
    if all(count >= SAMPLES_PER_CATEGORY_APPEARANCE for count in category_counts.values()):
        break
    if s['categories'] is None or s['abstract'] is None:
        continue
    
    current_categories = s['categories'].strip().split(' ')
    parent_categories = {cat.split('.')[0] for cat in current_categories}
    
    # Check if the sample contains at least one of our target categories
    found_target = False
    for p_cat in parent_categories:
        if p_cat in CATEGORIES_TO_SELECT and category_counts[p_cat] < SAMPLES_PER_CATEGORY_APPEARANCE:
            s['parent_categories'] = parent_categories
            samples.append(s)
            # Increment count for all found target categories in this sample
            for cat_to_increment in parent_categories:
                if cat_to_increment in category_counts:
                    category_counts[cat_to_increment] += 1
            break # Move to next sample once it's been added

print(f"Finished sampling. Total samples collected: {len(samples)}")
abstracts = [sample['abstract'] for sample in samples]
labels_sets = [sample['parent_categories'] for sample in samples]
processed_abstracts = [clean_text(abstract) for abstract in tqdm(abstracts, desc="Cleaning Abstracts")]

# Create multi-label indicator matrix Y
Y = np.zeros((len(samples), len(CATEGORIES_TO_SELECT)), dtype=int)
cat_to_idx = {cat: i for i, cat in enumerate(CATEGORIES_TO_SELECT)}
for i, label_set in enumerate(labels_sets):
    for label in label_set:
        if label in cat_to_idx:
            Y[i, cat_to_idx[label]] = 1

train_texts, test_texts, Y_train, Y_test = train_test_split(
    processed_abstracts, Y, test_size=0.2, random_state=RANDOM_STATE
)

# 2. Feature Engineering
print("\n--- Step 2: Enhanced Feature Engineering ---")
# Advanced TF-IDF
print("Creating Enhanced TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, min_df=5, max_df=0.7, sublinear_tf=True, ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
X_test_tfidf = tfidf_vectorizer.transform(test_texts)
# SBERT e5-base Embeddings
print(f"Creating SBERT Embeddings using {E5_MODEL_NAME}...")
sbert_model = SentenceTransformer(E5_MODEL_NAME, device=DEVICE)
X_train_emb = sbert_model.encode(train_texts, batch_size=BATCH_SIZE, show_progress_bar=True)
X_test_emb = sbert_model.encode(test_texts, batch_size=BATCH_SIZE, show_progress_bar=True)
print("All feature sets created.")

# --- Initialize empty results dictionary ---
results = {}

# --- TIER 1: BINARY RELEVANCE WITH LOGISTIC REGRESSION ---
print("\n--- Tier 1: Training Binary Relevance with LR(tfidf) ---")
log_message("\n\n--- Tier 1: Binary Relevance with LR(tfidf) ---")
lr_classifiers = {}
for i, category in enumerate(CATEGORIES_TO_SELECT):
    print(f"  Training classifier for: {category}")
    y_binary_train = Y_train[:, i]
    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced')
    model.fit(X_train_tfidf, y_binary_train)
    lr_classifiers[category] = model

# Predict and evaluate
Y_pred_lr = np.zeros_like(Y_test)
for i, category in enumerate(CATEGORIES_TO_SELECT):
    Y_pred_lr[:, i] = lr_classifiers[category].predict(X_test_tfidf)

results['BR_LR(tfidf)_accuracy'] = accuracy_score(Y_test, Y_pred_lr) # Subset accuracy
results['BR_LR(tfidf)_hamming'] = hamming_loss(Y_test, Y_pred_lr)
log_message(f"Overall Subset Accuracy: {results['BR_LR(tfidf)_accuracy']:.4f}")
log_message(f"Hamming Loss: {results['BR_LR(tfidf)_hamming']:.4f}\n")
log_message("Per-Category Performance:")
log_message(classification_report(Y_test, Y_pred_lr, target_names=CATEGORIES_TO_SELECT, zero_division=0))


# --- TIER 2: BINARY RELEVANCE WITH SOFT VOTING ENSEMBLE ---
print("\n--- Tier 2: Training Binary Relevance with Soft Voting Ensembles ---")
log_message("\n\n--- Tier 2: Binary Relevance with Soft Voting Ensembles ---")
voting_classifiers = {}
for i, category in enumerate(CATEGORIES_TO_SELECT):
    print(f"  Training SOFT VOTING ensemble for: {category}")
    y_binary_train = Y_train[:, i]
    
    # Define and train base models for this binary task
    mnb = MultinomialNB(alpha=0.1).fit(X_train_tfidf, y_binary_train)
    knn = KNeighborsClassifier(n_neighbors=7, weights='distance').fit(X_train_emb, y_binary_train)
    dt = DecisionTreeClassifier(max_depth=40, min_samples_leaf=1, random_state=RANDOM_STATE).fit(X_train_tfidf, y_binary_train)
    
    # Calibrate for better probabilities
    calibrated_knn = CalibratedClassifierCV(estimator=knn, cv=CV_FOLDS, method='isotonic').fit(X_train_emb, y_binary_train)
    calibrated_dt = CalibratedClassifierCV(estimator=dt, cv=CV_FOLDS, method='isotonic').fit(X_train_tfidf, y_binary_train)

    voting_classifiers[category] = {
        'mnb': mnb, 'knn': calibrated_knn, 'dt': calibrated_dt
    }

# Predict and evaluate
Y_pred_vote = np.zeros_like(Y_test)
for i, category in enumerate(CATEGORIES_TO_SELECT):
    models = voting_classifiers[category]
    mnb_probs = models['mnb'].predict_proba(X_test_tfidf)[:, 1]
    knn_probs = models['knn'].predict_proba(X_test_emb)[:, 1]
    dt_probs = models['dt'].predict_proba(X_test_tfidf)[:, 1]
    
    # Weighted average for the '1' class probability
    final_probs = (0.4 * mnb_probs) + (0.4 * knn_probs) + (0.2 * dt_probs)
    Y_pred_vote[:, i] = (final_probs >= 0.5).astype(int)

results['BR_SoftVote_accuracy'] = accuracy_score(Y_test, Y_pred_vote)
results['BR_SoftVote_hamming'] = hamming_loss(Y_test, Y_pred_vote)
log_message(f"Overall Subset Accuracy: {results['BR_SoftVote_accuracy']:.4f}")
log_message(f"Hamming Loss: {results['BR_SoftVote_hamming']:.4f}\n")
log_message("Per-Category Performance:")
log_message(classification_report(Y_test, Y_pred_vote, target_names=CATEGORIES_TO_SELECT, zero_division=0))


# --- TIER 3: BINARY RELEVANCE WITH STACKING ENSEMBLE ---
print("\n--- Tier 3: Training Binary Relevance with Stacking Ensembles (This will be slow) ---")
log_message("\n\n--- Tier 3: Binary Relevance with Stacking Ensembles ---")
stacking_classifiers = {}
for i, category in enumerate(CATEGORIES_TO_SELECT):
    print(f"  Training STACKING ensemble for: {category}")
    y_binary_train = Y_train[:, i]

    # Define base models
    mnb = MultinomialNB(alpha=0.1)
    knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
    dt = DecisionTreeClassifier(max_depth=40, min_samples_leaf=1, random_state=RANDOM_STATE)
    
    # Generate out-of-fold predictions for the meta-learner
    mnb_meta_train = cross_val_predict(mnb, X_train_tfidf, y_binary_train, cv=CV_FOLDS, method='predict_proba', n_jobs=-1)[:, 1]
    knn_meta_train = cross_val_predict(knn, X_train_emb, y_binary_train, cv=CV_FOLDS, method='predict_proba', n_jobs=-1)[:, 1]
    dt_meta_train = cross_val_predict(dt, X_train_tfidf, y_binary_train, cv=CV_FOLDS, method='predict_proba', n_jobs=-1)[:, 1]

    # Create meta-feature set
    meta_features_train = np.stack([mnb_meta_train, knn_meta_train, dt_meta_train], axis=1)
    meta_learner_train_X = hstack([csr_matrix(meta_features_train), X_train_tfidf]).tocsr()
    
    # Train meta-learner
    meta_learner = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced')
    meta_learner.fit(meta_learner_train_X, y_binary_train)

    # Train base models on full data for final prediction
    mnb.fit(X_train_tfidf, y_binary_train)
    knn.fit(X_train_emb, y_binary_train)
    dt.fit(X_train_tfidf, y_binary_train)

    stacking_classifiers[category] = {
        'mnb': mnb, 'knn': knn, 'dt': dt, 'meta_learner': meta_learner
    }

# Predict and evaluate
Y_pred_stack = np.zeros_like(Y_test)
for i, category in enumerate(CATEGORIES_TO_SELECT):
    models = stacking_classifiers[category]
    mnb_probs_test = models['mnb'].predict_proba(X_test_tfidf)[:, 1]
    knn_probs_test = models['knn'].predict_proba(X_test_emb)[:, 1]
    dt_probs_test = models['dt'].predict_proba(X_test_tfidf)[:, 1]

    meta_features_test = np.stack([mnb_probs_test, knn_probs_test, dt_probs_test], axis=1)
    meta_learner_test_X = hstack([csr_matrix(meta_features_test), X_test_tfidf]).tocsr()

    Y_pred_stack[:, i] = models['meta_learner'].predict(meta_learner_test_X)

results['BR_Stacking_accuracy'] = accuracy_score(Y_test, Y_pred_stack)
results['BR_Stacking_hamming'] = hamming_loss(Y_test, Y_pred_stack)
log_message(f"Overall Subset Accuracy: {results['BR_Stacking_accuracy']:.4f}")
log_message(f"Hamming Loss: {results['BR_Stacking_hamming']:.4f}\n")
log_message("Per-Category Performance:")
log_message(classification_report(Y_test, Y_pred_stack, target_names=CATEGORIES_TO_SELECT, zero_division=0))


# --- Final Summary ---
summary_header = f"\n\n--- Multi-Label Championship Summary ---"
table_header = f"{'Model Architecture':<45} | {'Subset Accuracy':<20} | {'Hamming Loss (Lower is Better)':<30}"
separator = "-" * len(table_header)
log_message(summary_header, to_console=True)
log_message(table_header, to_console=True)
log_message(separator, to_console=True)
# LR
row_str = f"{'Tier 1: Binary Relevance with LR(tfidf)':<45} | {results['BR_LR(tfidf)_accuracy']:<20.4f} | {results['BR_LR(tfidf)_hamming']:<30.4f}"
log_message(row_str, to_console=True)
# Soft Voting
row_str = f"{'Tier 2: Binary Relevance with Soft Voting':<45} | {results['BR_SoftVote_accuracy']:<20.4f} | {results['BR_SoftVote_hamming']:<30.4f}"
log_message(row_str, to_console=True)
# Stacking
row_str = f"{'Tier 3: Binary Relevance with Stacking':<45} | {results['BR_Stacking_accuracy']:<20.4f} | {results['BR_Stacking_hamming']:<30.4f}"
log_message(row_str, to_console=True)
log_message(separator, to_console=True)

print(f"\nMulti-label championship complete. Results appended to '{LOG_FILE_PATH}'.")