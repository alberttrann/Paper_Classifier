
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

# FAISS for kNN acceleration
import faiss

# Scikit-learn for models, vectorizers, and metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler

# Scipy for manual voting
from scipy.stats import mode
from tqdm.auto import tqdm
import torch

# --- Configuration ---
CATEGORIES_TO_SELECT = ['astro-ph', 'cond-mat', 'cs', 'math', 'physics']
SAMPLES_PER_CATEGORY = 2000 # Using 1000 for a robust result (5k total)
DATASET_NAME = "UniverseTBD/arxiv-abstracts-large"

MODEL_NAME = "intfloat/multilingual-e5-base"
TFIDF_MAX_FEATURES = 10000
KNN_N_NEIGHBORS = 5
DT_MAX_DEPTH = 20
RANDOM_STATE = 42
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_FILE_PATH = "benchmark_results.txt"

# --- NLTK Downloads ---
try:
    stopwords.words('english')
except LookupError:
    import nltk
    nltk.download('stopwords')
try:
    word_tokenize("test")
except LookupError:
    import nltk
    nltk.download('punkt')
try:
    WordNetLemmatizer().lemmatize("test")
except LookupError:
    import nltk
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# --- Helper function for logging ---
def log_message(message):
    """Prints a message to the console and appends it to the log file."""
    print(message)
    # Open in append mode ('a')
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
log_message(f"--- Heterogeneous Ensemble Benchmark Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
log_message("="*80)

# 1. Data Sampling and Preprocessing
print("--- Step 1: Data Sampling & Preprocessing ---")
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
# Bag of Words
bow_vectorizer = CountVectorizer(max_features=TFIDF_MAX_FEATURES)
X_train_bow = bow_vectorizer.fit_transform(X_train_text)
X_test_bow = bow_vectorizer.transform(X_test_text)
# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
# SBERT Embeddings
sbert_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
X_train_emb = sbert_model.encode(X_train_text, batch_size=BATCH_SIZE, show_progress_bar=True)
X_test_emb = sbert_model.encode(X_test_text, batch_size=BATCH_SIZE, show_progress_bar=True)
print("All feature sets created.")

# 3. Generate Base Model Predictions
print("\n--- Step 3: Generating Predictions from Base Models ---")
predictions = {}

# MNB Predictions
print("Generating MNB predictions (BoW & TF-IDF)...")
mnb_bow = MultinomialNB().fit(X_train_bow, y_train)
predictions['MNB_bow'] = mnb_bow.predict(X_test_bow)
mnb_tfidf = MultinomialNB().fit(X_train_tfidf, y_train)
predictions['MNB_tfidf'] = mnb_tfidf.predict(X_test_tfidf)

# kNN Predictions (only needs Embeddings)
print("Generating kNN predictions (Embeddings)...")
knn_index = faiss.IndexFlatL2(X_train_emb.shape[1])
knn_index.add(X_train_emb.astype(np.float32))
_, knn_indices = knn_index.search(X_test_emb.astype(np.float32), KNN_N_NEIGHBORS)
knn_neighbor_labels = y_train[knn_indices]
knn_preds, _ = mode(knn_neighbor_labels, axis=1)
predictions['kNN_emb'] = knn_preds.ravel()

# Decision Tree Predictions
print("Generating Decision Tree predictions (BoW & TF-IDF)...")
dt_bow = DecisionTreeClassifier(max_depth=DT_MAX_DEPTH, random_state=RANDOM_STATE).fit(X_train_bow, y_train)
predictions['DT_bow'] = dt_bow.predict(X_test_bow)
dt_tfidf = DecisionTreeClassifier(max_depth=DT_MAX_DEPTH, random_state=RANDOM_STATE).fit(X_train_tfidf, y_train)
predictions['DT_tfidf'] = dt_tfidf.predict(X_test_tfidf)
print("All base predictions generated.")

# 4. Define and Benchmark Ensembles
print("\n--- Step 4: Benchmarking Heterogeneous Ensembles ---")
ensemble_configs = {
    "Ensemble 1": ['MNB_bow', 'kNN_emb', 'DT_tfidf'],
    "Ensemble 2": ['MNB_tfidf', 'kNN_emb', 'DT_bow'],
    "Ensemble 3": ['MNB_tfidf', 'kNN_emb', 'DT_tfidf'],
    "Ensemble 4": ['MNB_bow', 'kNN_emb', 'DT_bow'],
    "Ensemble 5": ['MNB_bow', 'kNN_emb'],
    "Ensemble 6": ['MNB_tfidf', 'kNN_emb']
}
ensemble_accuracies = {}

log_message("\n\n--- Detailed Heterogeneous Ensemble Reports ---")

for name, components in ensemble_configs.items():
    component_names = " + ".join([c.replace('_', '(') + ')' for c in components])
    log_message("\n" + "="*50)
    log_message(f"Evaluating: {name} | Components: {component_names}")
    log_message("="*50)
    
    # Stack the pre-generated predictions
    preds_to_stack = [predictions[comp] for comp in components]
    stacked_preds = np.stack(preds_to_stack, axis=1)
    
    # Perform manual hard voting
    final_preds, _ = mode(stacked_preds, axis=1)
    final_preds = final_preds.ravel()
    
    # Calculate and store results
    accuracy = accuracy_score(y_test, final_preds)
    report = classification_report(y_test, final_preds, target_names=unique_labels, zero_division=0)
    
    ensemble_accuracies[name] = accuracy
    
    log_message(f"Overall Accuracy: {accuracy:.4f}\n")
    log_message(report)

# 5. Generate and Log Summary Table
summary_header = f"\n\n--- Heterogeneous Ensemble Summary (Accuracy) ---"
table_header = f"{'Ensemble Configuration':<50} | {'Accuracy':<15}"
separator = "-" * len(table_header)

# Log to file and print to console
log_message(summary_header)
log_message(table_header)
log_message(separator)
print(summary_header)
print(table_header)
print(separator)

for name, components in ensemble_configs.items():
    component_names = " + ".join([c.replace('_', '(') + ')' for c in components])
    accuracy = ensemble_accuracies[name]
    row_str = f"{component_names:<50} | {accuracy:<15.4f}"
    log_message(row_str)
    print(row_str)

log_message(separator)
print(separator)

print(f"\nBenchmark complete. All results have been appended to '{LOG_FILE_PATH}'.")