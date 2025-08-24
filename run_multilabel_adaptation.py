# run_multilabel_adaptation_benchmark.py

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
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, hamming_loss

# NEW: Import ClassifierChain from scikit-multilearn
from skmultilearn.problem_transform import ClassifierChain

from tqdm.auto import tqdm
import torch

# --- Configuration ---
# Data Sampling
CATEGORIES_TO_SELECT = [
    'math', 'astro-ph', 'cs', 'cond-mat', 'physics', 
    'hep-ph', 'quant-ph', 'hep-th'
]
SAMPLES_PER_CATEGORY_APPEARANCE = 5000 

# Models & Vectorizers
E5_MODEL_NAME = "intfloat/multilingual-e5-base"
TFIDF_MAX_FEATURES = 10000
RANDOM_STATE = 42
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_FILE_PATH = "multilabel_adaptation_benchmarks.txt"

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
log_message(f"--- Multi-Label Adaptation Method Benchmark (Corrected): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
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
    
    found_target = False
    for p_cat in parent_categories:
        if p_cat in CATEGORIES_TO_SELECT and category_counts[p_cat] < SAMPLES_PER_CATEGORY_APPEARANCE:
            s['parent_categories'] = parent_categories
            samples.append(s)
            for cat_to_increment in parent_categories:
                if cat_to_increment in category_counts:
                    category_counts[cat_to_increment] += 1
            break

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

# --- 3. Benchmarking Natively Multi-Label Models ---
log_message("\n\n--- Detailed Algorithm Adaptation Reports ---")

# a. RandomForest on TF-IDF using ClassifierChain
print("\n--- Benchmarking RandomForest on TF-IDF (using ClassifierChain) ---")
# Use ClassifierChain to properly adapt RandomForest for multi-label tasks
# This will train 8 RandomForest models in a sequence
chain_rf_tfidf = ClassifierChain(
    classifier = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, max_depth=40, class_weight='balanced'),
    require_dense = [False, True]
)
chain_rf_tfidf.fit(X_train_tfidf, Y_train)
Y_pred_rf_tfidf = chain_rf_tfidf.predict(X_test_tfidf)
results['RF_Chain(tfidf)_accuracy'] = accuracy_score(Y_test, Y_pred_rf_tfidf)
results['RF_Chain(tfidf)_hamming'] = hamming_loss(Y_test, Y_pred_rf_tfidf)
log_message("\n" + "="*50 + "\nModel: ClassifierChain(RandomForest(TF-IDF))\n" + "="*50)
log_message(f"Overall Subset Accuracy: {results['RF_Chain(tfidf)_accuracy']:.4f}")
log_message(f"Hamming Loss: {results['RF_Chain(tfidf)_hamming']:.4f}\n")
log_message("Per-Category Performance:")
log_message(classification_report(Y_test, Y_pred_rf_tfidf, target_names=CATEGORIES_TO_SELECT, zero_division=0))

# b. RandomForest on Embeddings using ClassifierChain
print("\n--- Benchmarking RandomForest on Embeddings (using ClassifierChain) ---")
chain_rf_emb = ClassifierChain(
    classifier = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, max_depth=40, class_weight='balanced'),
    require_dense = [False, True]
)
chain_rf_emb.fit(X_train_emb, Y_train)
Y_pred_rf_emb = chain_rf_emb.predict(X_test_emb)
results['RF_Chain(emb)_accuracy'] = accuracy_score(Y_test, Y_pred_rf_emb)
results['RF_Chain(emb)_hamming'] = hamming_loss(Y_test, Y_pred_rf_emb)
log_message("\n" + "="*50 + "\nModel: ClassifierChain(RandomForest(Embeddings))\n" + "="*50)
log_message(f"Overall Subset Accuracy: {results['RF_Chain(emb)_accuracy']:.4f}")
log_message(f"Hamming Loss: {results['RF_Chain(emb)_hamming']:.4f}\n")
log_message("Per-Category Performance:")
log_message(classification_report(Y_test, Y_pred_rf_emb, target_names=CATEGORIES_TO_SELECT, zero_division=0))


# --- Final Summary ---
summary_header = f"\n\n--- Multi-Label Adaptation Methods Summary ---"
table_header = f"{'Model Architecture':<45} | {'Subset Accuracy':<20} | {'Hamming Loss (Lower is Better)':<30}"
separator = "-" * len(table_header)
log_message(summary_header, to_console=True)
log_message(table_header, to_console=True)
log_message(separator, to_console=True)

# RandomForest TFIDF
row_str = f"{'ClassifierChain(RandomForest(TF-IDF))':<45} | {results['RF_Chain(tfidf)_accuracy']:<20.4f} | {results['RF_Chain(tfidf)_hamming']:<30.4f}"
log_message(row_str, to_console=True)
# RandomForest Embeddings
row_str = f"{'ClassifierChain(RandomForest(Embeddings))':<45} | {results['RF_Chain(emb)_accuracy']:<20.4f} | {results['RF_Chain(emb)_hamming']:<30.4f}"
log_message(row_str, to_console=True)

log_message(separator, to_console=True)
print(f"\nMulti-label adaptation benchmark complete. Results appended to '{LOG_FILE_PATH}'.")