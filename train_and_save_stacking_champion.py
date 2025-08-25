# train_and_save_stacking_champion.py

import os
import re
import string
import numpy as np
from datasets import load_dataset
import joblib

# NLTK
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Sentence Transformers
from sentence_transformers import SentenceTransformer

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# Other
from scipy.sparse import hstack, issparse
from tqdm.auto import tqdm
import torch

# --- Configuration ---
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
OUTPUT_MODEL_DIR = "stacking_champion_model"

# --- NLTK Downloads & Text Preprocessing (Same as before) ---
# ... (Assuming NLTK data is downloaded) ...
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
domain_specific_stopwords = {'result', 'study', 'show', 'paper', 'model', 'analysis', 'method', 'approach', 'propose', 'demonstrate', 'investigate', 'present'}
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
print(f"--- Training and Saving the Stacking Champion Model ---")
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

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
        samples.append({'abstract': s['abstract'], 'category': parent_category})
        category_counts[parent_category] += 1
print(f"Finished sampling. Total samples collected: {len(samples)}")
abstracts = [s['abstract'] for s in samples]
labels_str = [s['category'] for s in samples]
processed_abstracts = [clean_text(abstract) for abstract in tqdm(abstracts, desc="Cleaning Abstracts")]
unique_labels = sorted(list(set(labels_str)))
label_to_int = {label: i for i, label in enumerate(unique_labels)}
labels = np.array([label_to_int[label] for label in labels_str])
train_texts, _, y_train, _ = train_test_split(
    processed_abstracts, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
)
# Save the label mapping for later use
joblib.dump(unique_labels, os.path.join(OUTPUT_MODEL_DIR, 'unique_labels.pkl'))

# 2. Feature Engineering
print("\n--- Step 2: Enhanced Feature Engineering ---")
# Advanced TF-IDF
print("Creating and saving Enhanced TF-IDF vectorizer...")
tfidf_vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, min_df=5, max_df=0.7, sublinear_tf=True, ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
joblib.dump(tfidf_vectorizer, os.path.join(OUTPUT_MODEL_DIR, 'tfidf_vectorizer.pkl'))
# SBERT e5-base Embeddings
print(f"Creating SBERT Embeddings using {E5_MODEL_NAME}...")
sbert_model = SentenceTransformer(E5_MODEL_NAME, device=DEVICE)
X_train_emb = sbert_model.encode(train_texts, batch_size=BATCH_SIZE, show_progress_bar=True)
print("Feature sets created.")

# 3. Hyperparameter Tuning of Base Models
print("\n--- Step 3: Hyperparameter Tuning of Base Models ---")
# MNB
print("Tuning and saving MultinomialNB...")
mnb_params = {'alpha': [0.01, 0.1, 0.5, 1.0]}
mnb_grid = GridSearchCV(MultinomialNB(), mnb_params, cv=CV_FOLDS, n_jobs=-1)
mnb_grid.fit(X_train_tfidf, y_train)
best_mnb = mnb_grid.best_estimator_
joblib.dump(best_mnb, os.path.join(OUTPUT_MODEL_DIR, 'base_model_mnb.pkl'))
print(f"Best MNB params: {mnb_grid.best_params_}")
# kNN
print("Tuning and saving KNeighborsClassifier...")
knn_params = {'n_neighbors': [5, 7, 11], 'weights': ['uniform', 'distance']}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=CV_FOLDS, n_jobs=-1)
knn_grid.fit(X_train_emb, y_train)
best_knn = knn_grid.best_estimator_
joblib.dump(best_knn, os.path.join(OUTPUT_MODEL_DIR, 'base_model_knn.pkl'))
print(f"Best kNN params: {knn_grid.best_params_}")
# DT
print("Tuning and saving DecisionTreeClassifier...")
dt_params = {'max_depth': [20, 30, 40], 'min_samples_leaf': [1, 5, 10]}
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_STATE), dt_params, cv=CV_FOLDS, n_jobs=-1)
dt_grid.fit(X_train_tfidf, y_train)
best_dt = dt_grid.best_estimator_
joblib.dump(best_dt, os.path.join(OUTPUT_MODEL_DIR, 'base_model_dt.pkl'))
print(f"Best DT params: {dt_grid.best_params_}")

# 4. Generate Out-of-Fold Predictions for Stacking Meta-Learner
print("\n--- Step 4: Generating Meta-Features ---")
calibrated_dt = CalibratedClassifierCV(estimator=best_dt, cv=CV_FOLDS, method='isotonic')
calibrated_knn = CalibratedClassifierCV(estimator=best_knn, cv=CV_FOLDS, method='isotonic')
base_models_for_stacking = {'MNB_tfidf': best_mnb, 'kNN_emb': calibrated_knn, 'DT_tfidf': calibrated_dt}
meta_features_train = {}
for name, model in base_models_for_stacking.items():
    feature_type = name.split('_')[1]
    X_for_model = X_train_tfidf if feature_type == 'tfidf' else X_train_emb
    meta_features_train[name] = cross_val_predict(model, X_for_model, y_train, cv=CV_FOLDS, method='predict_proba', n_jobs=-1)

# 5. Train the Final Meta-Learner
print("\n--- Step 5: Training and Saving the Final Meta-Learner ---")
# Create the full training set for the meta-learner
base_meta_features_train = [meta_features_train['MNB_tfidf'], meta_features_train['kNN_emb'], meta_features_train['DT_tfidf']]
meta_learner_train_X = hstack(base_meta_features_train + [X_train_tfidf]).tocsr()
# Define and train the meta-learner
meta_learner = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
meta_learner.fit(meta_learner_train_X, y_train)
joblib.dump(meta_learner, os.path.join(OUTPUT_MODEL_DIR, 'meta_learner_lr.pkl'))
print("Meta-learner trained and saved.")

print(f"\nChampion stacking model and all its components have been trained and saved to the '{OUTPUT_MODEL_DIR}' directory.")