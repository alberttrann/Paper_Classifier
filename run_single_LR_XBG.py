
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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# XGBoost for a powerful single model
import xgboost as xgb

from tqdm.auto import tqdm
import torch

# --- Configuration ---
# Data Sampling
CATEGORIES_TO_SELECT = ['astro-ph', 'cond-mat', 'cs', 'math', 'physics']
SAMPLES_PER_CATEGORY = 2000 # Use 2000 for a robust result (10k total)

# Models & Vectorizers
E5_MODEL_NAME = "intfloat/multilingual-e5-base" # Using the confirmed best embedding model
TFIDF_MAX_FEATURES = 10000
RANDOM_STATE = 42
BATCH_SIZE = 128
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

# --- Main Execution ---
log_message("\n\n" + "="*80)
log_message(f"--- Single Model Benchmark (LR & XGBoost): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
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
processed_abstracts = [clean_text(abstract) for abstract in tqdm(abstracts, desc="Cleaning Text")]
unique_labels = sorted(list(set(labels_str)))
label_to_int = {label: i for i, label in enumerate(unique_labels)}
labels = np.array([label_to_int[label] for label in labels_str])
train_texts, test_texts, y_train, y_test = train_test_split(
    processed_abstracts, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
)

# 2. Feature Engineering
print("\n--- Step 2: Feature Engineering (All types) ---")
# Enhanced Bag of Words
print("Creating Enhanced Bag of Words features...")
bow_vectorizer = CountVectorizer(
    max_features=TFIDF_MAX_FEATURES, min_df=5, max_df=0.7, 
    ngram_range=(1, 2)
)
X_train_bow = bow_vectorizer.fit_transform(train_texts)
X_test_bow = bow_vectorizer.transform(test_texts)
# Enhanced TF-IDF
print("Creating Enhanced TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES, min_df=5, max_df=0.7, 
    sublinear_tf=True, ngram_range=(1, 2)
)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
X_test_tfidf = tfidf_vectorizer.transform(test_texts)
# SBERT e5-base Embeddings
print(f"Creating SBERT Embeddings using {E5_MODEL_NAME}...")
sbert_model = SentenceTransformer(E5_MODEL_NAME, device=DEVICE)
X_train_emb = sbert_model.encode(train_texts, batch_size=BATCH_SIZE, show_progress_bar=True)
X_test_emb = sbert_model.encode(test_texts, batch_size=BATCH_SIZE, show_progress_bar=True)
print("All feature sets created.")

# 3. Define Benchmark Configurations
print("\n--- Step 3: Running Single Model Benchmarks ---")
benchmark_configs = {
    "LR(BoW)":     {'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000), 'X_train': X_train_bow, 'X_test': X_test_bow},
    "LR(TFIDF)":   {'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000), 'X_train': X_train_tfidf, 'X_test': X_test_tfidf},
    "LR(Emb)":     {'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000), 'X_train': X_train_emb, 'X_test': X_test_emb},
    "XGB(BoW)":    {'model': xgb.XGBClassifier(random_state=RANDOM_STATE), 'X_train': X_train_bow, 'X_test': X_test_bow},
    "XGB(TFIDF)":  {'model': xgb.XGBClassifier(random_state=RANDOM_STATE), 'X_train': X_train_tfidf, 'X_test': X_test_tfidf},
    "XGB(Emb)":    {'model': xgb.XGBClassifier(random_state=RANDOM_STATE), 'X_train': X_train_emb, 'X_test': X_test_emb},
}
benchmark_accuracies = {}

log_message("\n\n--- Detailed Single Model Reports (LR & XGBoost) ---")

for name, config in benchmark_configs.items():
    model = config['model']
    X_train = config['X_train']
    X_test = config['X_test']
    
    log_message("\n" + "="*50)
    log_message(f"Evaluating Model: {name}")
    log_message("="*50)
    
    print(f"  Training {name}...")
    model.fit(X_train, y_train)
    
    print(f"  Predicting with {name}...")
    final_preds = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, final_preds)
    report = classification_report(y_test, final_preds, target_names=unique_labels, zero_division=0)
    
    benchmark_accuracies[name] = accuracy
    
    log_message(f"Overall Accuracy: {accuracy:.4f}\n" + report)

# 4. Generate and Log Summary Table
summary_header = f"\n\n--- Single Model (LR & XGBoost) Summary ---"
table_header = f"{'Model Configuration':<25} | {'Accuracy':<15}"
separator = "-" * len(table_header)

log_message(summary_header, to_console=True)
log_message(table_header, to_console=True)
log_message(separator, to_console=True)

for name, accuracy in benchmark_accuracies.items():
    row_str = f"{name:<25} | {accuracy:<15.4f}"
    log_message(row_str, to_console=True)

log_message(separator, to_console=True)
print(f"\nSingle model benchmark complete. Results appended to '{LOG_FILE_PATH}'.")