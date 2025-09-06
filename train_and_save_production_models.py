# train_and_save_production_models.py

import os
import re
import string
import numpy as np
import joblib
from datasets import load_dataset
from collections import Counter

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Sentence Transformers
from sentence_transformers import SentenceTransformer

# Sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline # Important!

from scipy.sparse import hstack, csr_matrix, issparse
from tqdm.auto import tqdm
import torch

print("--- Starting Production Model Training Script ---")
print("This will train and save all necessary models for the Streamlit app.")

# --- Configuration (Same as our best single-label run) ---
CATEGORIES_TO_SELECT = [
    'math', 'astro-ph', 'cs', 'cond-mat', 'physics', 
    'hep-ph', 'quant-ph', 'hep-th'
]
SAMPLES_PER_CATEGORY = 5000
E5_MODEL_NAME = "intfloat/multilingual-e5-base"
TFIDF_MAX_FEATURES = 10000
RANDOM_STATE = 42
CV_FOLDS = 5
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "production_models"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- NLTK Setup & Text Preprocessing ---
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
domain_specific_stopwords = {'result', 'study', 'show', 'paper', 'model', 'analysis', 'method', 'approach', 'propose', 'demonstrate'}
stop_words.update(domain_specific_stopwords)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(cleaned_tokens)

# --- 1. Data Sampling and Splitting (Single-Label) ---
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
        samples.append({'abstract': s['abstract'], 'category': parent_category})
        category_counts[parent_category] += 1
print(f"Finished sampling. Total samples collected: {len(samples)}")
abstracts = [s['abstract'] for s in samples]
labels_str = [s['category'] for s in samples]
processed_abstracts = [clean_text(abstract) for abstract in tqdm(abstracts, desc="Cleaning Abstracts")]
unique_labels = sorted(list(set(labels_str)))
label_to_int = {label: i for i, label in enumerate(unique_labels)}
labels = np.array([label_to_int[label] for label in labels_str])
# Save the final label list
joblib.dump(unique_labels, os.path.join(OUTPUT_DIR, 'final_labels.pkl'))

# Create the full training set (no test split needed, since we're just saving models)
X_train_text = processed_abstracts
y_train = labels

# --- 2. Feature Engineering ---
print("\n--- Step 2: Enhanced Feature Engineering ---")
print("Fitting and saving Enhanced TF-IDF vectorizer...")
tfidf_vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, min_df=5, max_df=0.7, sublinear_tf=True, ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
joblib.dump(tfidf_vectorizer, os.path.join(OUTPUT_DIR, 'tfidf_vectorizer.pkl'))

print(f"Creating and saving SBERT e5-base model...")
sbert_model = SentenceTransformer(E5_MODEL_NAME, device=DEVICE)
X_train_emb = sbert_model.encode(X_train_text, batch_size=BATCH_SIZE, show_progress_bar=True)
# Save the model itself. The model is saved, not just the embeddings.
sbert_model.save(os.path.join(OUTPUT_DIR, 'e5_embedding_model'))
print("Feature sets created and models saved.")

# --- 3. Train and Save Models ---

# Model 1: Best Single Model (LR on TFIDF)
print("\n--- Training and saving Best Single Model: LR(tfidf) ---")
lr_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
joblib.dump(lr_model, os.path.join(OUTPUT_DIR, 'model_lr_tfidf.pkl'))

# Models 2 & 3: Soft Voting and Stacking Base Models
print("\n--- Training and saving Base Models for Ensembles ---")
# Tuned MNB
print("Tuning and saving MNB...")
mnb_params = {'alpha': [0.01, 0.1, 0.5, 1.0]}
mnb_grid = GridSearchCV(MultinomialNB(), mnb_params, cv=CV_FOLDS, n_jobs=-1)
mnb_grid.fit(X_train_tfidf, y_train)
best_mnb = mnb_grid.best_estimator_
joblib.dump(best_mnb, os.path.join(OUTPUT_DIR, 'base_model_mnb.pkl'))
# Tuned kNN (Calibrated)
print("Tuning, calibrating, and saving kNN...")
knn_params = {'n_neighbors': [7], 'weights': ['distance']} # Using our known best params
best_knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
calibrated_knn = CalibratedClassifierCV(estimator=best_knn, cv=CV_FOLDS, method='isotonic')
calibrated_knn.fit(X_train_emb, y_train)
joblib.dump(calibrated_knn, os.path.join(OUTPUT_DIR, 'base_model_knn_calibrated.pkl'))
# Tuned DT (Calibrated)
print("Tuning, calibrating, and saving DT...")
dt_params = {'max_depth': [40], 'min_samples_leaf': [1]} # Using our known best params
best_dt = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=40, min_samples_leaf=1)
calibrated_dt = CalibratedClassifierCV(estimator=best_dt, cv=CV_FOLDS, method='isotonic')
calibrated_dt.fit(X_train_tfidf, y_train)
joblib.dump(calibrated_dt, os.path.join(OUTPUT_DIR, 'base_model_dt_calibrated.pkl'))

# Model 4: Stacking Champion Meta-Learner
print("\n--- Training and saving Stacking Meta-Learner ---")
print("Generating out-of-fold predictions...")
meta_features_train = {}
meta_features_train['MNB_tfidf'] = cross_val_predict(best_mnb, X_train_tfidf, y_train, cv=CV_FOLDS, method='predict_proba', n_jobs=-1)
meta_features_train['kNN_emb'] = cross_val_predict(best_knn, X_train_emb, y_train, cv=CV_FOLDS, method='predict_proba', n_jobs=-1) # Use uncalibrated for CV stacking
meta_features_train['DT_tfidf'] = cross_val_predict(best_dt, X_train_tfidf, y_train, cv=CV_FOLDS, method='predict_proba', n_jobs=-1)
# Create the full training set for the meta-learner
base_meta_features_train = [meta_features_train['MNB_tfidf'], meta_features_train['kNN_emb'], meta_features_train['DT_tfidf']]
meta_learner_train_X = hstack(base_meta_features_train + [X_train_tfidf]).tocsr()
# Define and train the meta-learner
meta_learner = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
meta_learner.fit(meta_learner_train_X, y_train)
joblib.dump(meta_learner, os.path.join(OUTPUT_DIR, 'stacking_meta_learner.pkl'))
print("Meta-learner trained and saved.")

print("\n--- ALL PRODUCTION MODELS TRAINED AND SAVED. ---")