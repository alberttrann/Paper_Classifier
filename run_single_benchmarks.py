# run_benchmark.py

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
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler

from tqdm.auto import tqdm
import torch

# --- Configuration ---
# Data Sampling
CATEGORIES_TO_SELECT = ['astro-ph', 'cond-mat', 'cs', 'math', 'physics']
SAMPLES_PER_CATEGORY = 1000 # Using 1000 per category for a more robust benchmark (5k total)
DATASET_NAME = "UniverseTBD/arxiv-abstracts-large"  # Replace with your dataset name

# Models & Vectorizers
MODEL_NAME = "intfloat/multilingual-e5-base"
TFIDF_MAX_FEATURES = 10000 # Limit TF-IDF/BoW features to prevent excessive memory usage
KNN_N_NEIGHBORS = 5 # A common choice for k
DT_MAX_DEPTH = 20 # A reasonable depth to prevent severe overfitting
RANDOM_STATE = 42
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# NEW: Log file configuration
LOG_FILE_PATH = "benchmark_results.txt"

# --- NLTK Downloads (run once if not already downloaded) ---
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

# --- Model Training & Testing Functions ---

def train_and_test_faiss_knn(X_train, y_train, X_test, y_test, n_neighbors):
    print(f"  Training FAISS index for kNN (k={n_neighbors})...")
    index = faiss.IndexFlatL2(X_train.shape[1])
    index.add(X_train.astype(np.float32))
    
    print("  Predicting with kNN...")
    _, indices = index.search(X_test.astype(np.float32), n_neighbors)
    neighbor_labels = y_train[indices]
    
    from scipy.stats import mode
    predictions, _ = mode(neighbor_labels, axis=1)
    
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=False, zero_division=0) # Get string report
    return accuracy, report

def train_and_test_mnb(X_train, y_train, X_test, y_test, is_embedding=False):
    print("  Training Multinomial Naive Bayes...")
    # MNB requires non-negative features. Embeddings can have negative values.
    if is_embedding:
        print("    (Applying MinMaxScaler for embeddings)")
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    print("  Predicting with MNB...")
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=False, zero_division=0) # Get string report
    return accuracy, report

def train_and_test_dt(X_train, y_train, X_test, y_test, max_depth):
    print(f"  Training Decision Tree (max_depth={max_depth})...")
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    
    print("  Predicting with Decision Tree...")
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=False, zero_division=0) # Get string report
    return accuracy, report

def train_and_test_kmeans(X_train, y_train, X_test, y_test, n_clusters):
    """ K-Means for classification, based on the provided teacher's code. """
    print(f"  Training K-Means (n_clusters={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    cluster_ids_train = kmeans.fit_predict(X_train)
    
    # Assign a class label to each cluster
    cluster_to_label = {}
    for cluster_id in set(cluster_ids_train):
        labels_in_cluster = [y_train[i] for i in range(len(y_train)) if cluster_ids_train[i] == cluster_id]
        if not labels_in_cluster:
            # Handle empty clusters by assigning a default label (e.g., the most common overall)
            most_common_label = Counter(y_train).most_common(1)[0][0]
        else:
            most_common_label = Counter(labels_in_cluster).most_common(1)[0][0]
        cluster_to_label[cluster_id] = most_common_label
        
    print("  Predicting with K-Means...")
    cluster_ids_test = kmeans.predict(X_test)
    
    # Map test cluster IDs to the learned labels
    # Use a default label for any unforeseen cluster IDs
    default_label = Counter(y_train).most_common(1)[0][0]
    predictions = [cluster_to_label.get(cluster_id, default_label) for cluster_id in cluster_ids_test]
    
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=False, zero_division=0) # Get string report
    return accuracy, report


# --- Main Execution ---

# Initialize log file
if os.path.exists(LOG_FILE_PATH):
    os.remove(LOG_FILE_PATH) # Remove old log file to start fresh
log_message(f"--- Benchmark Run Initiated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
log_message("\n--- Configuration ---")
log_message(f"Categories: {CATEGORIES_TO_SELECT}")
log_message(f"Samples per Category: {SAMPLES_PER_CATEGORY}")
log_message(f"SBERT Model: {MODEL_NAME}")
log_message(f"Device: {DEVICE}")
log_message("-" * 25 + "\n")


# 1. Data Sampling
print("--- Step 1: Data Sampling ---")
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

# 2. Text Preprocessing
print("\n--- Step 2: Text Preprocessing ---")
abstracts = [sample['abstract'] for sample in samples]
labels_str = [sample['parent_category'] for sample in samples]
processed_abstracts = [clean_text(abstract) for abstract in tqdm(abstracts, desc="Cleaning Text")]
unique_labels = sorted(list(set(labels_str)))
label_to_int = {label: i for i, label in enumerate(unique_labels)}
labels = np.array([label_to_int[label] for label in labels_str])
X_train_text, X_test_text, y_train, y_test = train_test_split(
    processed_abstracts, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
)

# 3. Feature Engineering (Vectorization)
print("\n--- Step 3: Feature Engineering ---")

# Bag of Words
print("Creating Bag of Words features...")
bow_vectorizer = CountVectorizer(max_features=TFIDF_MAX_FEATURES)
X_train_bow = bow_vectorizer.fit_transform(X_train_text)
X_test_bow = bow_vectorizer.transform(X_test_text)

# TF-IDF
print("Creating TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

# SBERT Embeddings
print(f"Creating SBERT Embeddings using {MODEL_NAME}...")
sbert_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
X_train_emb = sbert_model.encode(X_train_text, batch_size=BATCH_SIZE, show_progress_bar=True)
X_test_emb = sbert_model.encode(X_test_text, batch_size=BATCH_SIZE, show_progress_bar=True)

# 4. Benchmarking
print("\n--- Step 4: Running Benchmarks ---")
results_acc = {}
results_report = {}

# kNN Benchmark
print("\nBenchmarking kNN...")
results_acc['kNN_bow'], results_report['kNN_bow'] = train_and_test_faiss_knn(X_train_bow.toarray(), y_train, X_test_bow.toarray(), y_test, KNN_N_NEIGHBORS)
results_acc['kNN_tfidf'], results_report['kNN_tfidf'] = train_and_test_faiss_knn(X_train_tfidf.toarray(), y_train, X_test_tfidf.toarray(), y_test, KNN_N_NEIGHBORS)
results_acc['kNN_emb'], results_report['kNN_emb'] = train_and_test_faiss_knn(X_train_emb, y_train, X_test_emb, y_test, KNN_N_NEIGHBORS)

# MultinomialNB Benchmark
print("\nBenchmarking MultinomialNB...")
results_acc['MNB_bow'], results_report['MNB_bow'] = train_and_test_mnb(X_train_bow, y_train, X_test_bow, y_test)
results_acc['MNB_tfidf'], results_report['MNB_tfidf'] = train_and_test_mnb(X_train_tfidf, y_train, X_test_tfidf, y_test)
results_acc['MNB_emb'], results_report['MNB_emb'] = train_and_test_mnb(X_train_emb, y_train, X_test_emb, y_test, is_embedding=True)

# Decision Tree Benchmark
print("\nBenchmarking Decision Tree...")
results_acc['DT_bow'], results_report['DT_bow'] = train_and_test_dt(X_train_bow, y_train, X_test_bow, y_test, DT_MAX_DEPTH)
results_acc['DT_tfidf'], results_report['DT_tfidf'] = train_and_test_dt(X_train_tfidf, y_train, X_test_tfidf, y_test, DT_MAX_DEPTH)
results_acc['DT_emb'], results_report['DT_emb'] = train_and_test_dt(X_train_emb, y_train, X_test_emb, y_test, DT_MAX_DEPTH)

# K-Means Benchmark
print("\nBenchmarking K-Means...")
n_clusters = len(unique_labels)
results_acc['KMeans_bow'], results_report['KMeans_bow'] = train_and_test_kmeans(X_train_bow, y_train, X_test_bow, y_test, n_clusters)
results_acc['KMeans_tfidf'], results_report['KMeans_tfidf'] = train_and_test_kmeans(X_train_tfidf, y_train, X_test_tfidf, y_test, n_clusters)
results_acc['KMeans_emb'], results_report['KMeans_emb'] = train_and_test_kmeans(X_train_emb, y_train, X_test_emb, y_test, n_clusters)


# 5. Log Detailed Reports
log_message("\n\n--- Detailed Classification Reports ---")
for model in ['kNN', 'MNB', 'DT', 'KMeans']:
    for feature_type in ['bow', 'tfidf', 'emb']:
        key = f"{model}_{feature_type}"
        feature_name = {"bow": "Bag of Words", "tfidf": "TF-IDF", "emb": "Embeddings"}[feature_type]
        
        log_message("\n" + "="*50)
        log_message(f"Model: {model} with Features: {feature_name}")
        log_message("="*50)
        log_message(f"Overall Accuracy: {results_acc.get(key, 0):.4f}\n")
        log_message(results_report.get(key, "Report not generated."))


# 6. Generate and Log Summary Table
summary_header = f"\n\n--- Benchmark Results Summary (Accuracy) ---"
table_header = f"{'Algorithm':<15} | {'Bag of Words':<15} | {'TF-IDF':<15} | {'Embeddings':<15}"
separator = "-" * len(table_header)

# Log to file
log_message(summary_header)
log_message(table_header)
log_message(separator)

# Print to console
print(summary_header)
print(table_header)
print(separator)

for model in ['kNN', 'MNB', 'DT', 'KMeans']:
    bow_acc = results_acc.get(f'{model}_bow', 0)
    tfidf_acc = results_acc.get(f'{model}_tfidf', 0)
    emb_acc = results_acc.get(f'{model}_emb', 0)
    
    row_str = f"{model:<15} | {bow_acc:<15.4f} | {tfidf_acc:<15.4f} | {emb_acc:<15.4f}"
    log_message(row_str)
    print(row_str)

log_message(separator)
print(separator)

print(f"\nBenchmark complete. All results have been logged to '{LOG_FILE_PATH}'.")