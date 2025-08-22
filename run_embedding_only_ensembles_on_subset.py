
import os
import re
import string
import numpy as np
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import faiss
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import mode
from tqdm.auto import tqdm
import torch

# --- Configuration ---
# Define the parent categories to focus on and the number of samples per category
CATEGORIES_TO_SELECT = ['astro-ph', 'cond-mat', 'cs', 'math', 'physics']
SAMPLES_PER_CATEGORY = 2000 
TOTAL_SAMPLES = len(CATEGORIES_TO_SELECT) * SAMPLES_PER_CATEGORY

DATASET_NAME = "UniverseTBD/arxiv-abstracts-large"
MODEL_NAME = "intfloat/multilingual-e5-base"
OUTPUT_DIR = "processed_data_subset" # Use a new directory for this smaller dataset
KNN_N_NEIGHBORS = 5
RANDOM_STATE = 42
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- NLTK Downloads  ---
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

# --- Text Preprocessing Function  ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF" u"\U00002702-\U000027B0" u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(cleaned_tokens)

# --- Main Execution ---
print(f"Using device: {DEVICE}")

# --- Targeted Data Sampling ---
print(f"Starting targeted data sampling...")
print(f"Goal: {SAMPLES_PER_CATEGORY} samples for each of the following categories: {CATEGORIES_TO_SELECT}")

# Use a dictionary to track counts for each category
category_counts = {cat: 0 for cat in CATEGORIES_TO_SELECT}
samples = []

# Load dataset in streaming mode to efficiently iterate through it
dataset_generator = load_dataset(DATASET_NAME, split="train", streaming=True)

for s in tqdm(dataset_generator, desc="Scanning for samples"):
    # Stop when we have enough samples for all categories
    if all(count >= SAMPLES_PER_CATEGORY for count in category_counts.values()):
        break

    # Skip if categories or abstract is missing
    if s['categories'] is None or s['abstract'] is None:
        continue

    if len(s['categories'].split(' ')) != 1:
        continue

    # Extract the parent category (e.g., 'cs' from 'cs.LG')
    parent_category = s['categories'].strip().split('.')[0]

    # Check if this is a category we want and if we still need samples for it
    if parent_category in CATEGORIES_TO_SELECT and category_counts[parent_category] < SAMPLES_PER_CATEGORY:
        # Add the parent category to the sample dictionary for easy access
        s['parent_category'] = parent_category
        samples.append(s)
        category_counts[parent_category] += 1

print(f"Finished sampling. Total samples collected: {len(samples)}")
for category, count in category_counts.items():
    print(f"  - {category}: {count} samples")

# --- 2. Preprocess Text and Generate Embeddings for the Subset ---
print("\nPreprocessing text for the selected samples...")
abstracts = [sample['abstract'] for sample in samples]
labels_str = [sample['parent_category'] for sample in samples]
processed_abstracts = [clean_text(abstract) for abstract in tqdm(abstracts, desc="Cleaning Text")]

print(f"Loading SentenceTransformer model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)
model.eval()

print("Generating embeddings for the subset...")
embeddings = model.encode(processed_abstracts,
                          batch_size=BATCH_SIZE,
                          show_progress_bar=True,
                          convert_to_numpy=True,
                          device=DEVICE)

# --- 3. Final Unified Preprocessing (L2-Norm + MinMaxScaler) ---
# This ensures all models receive data in a compatible format
print("Applying final preprocessing to embeddings (L2-Norm + MinMaxScaler)...")
embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
scaler = MinMaxScaler()
final_embeddings = scaler.fit_transform(embeddings_normalized)
print(f"Final preprocessed embeddings shape: {final_embeddings.shape}")

# --- 4. Map String Labels to Integers ---
unique_labels = sorted(list(set(labels_str)))
label_to_int = {label: i for i, label in enumerate(unique_labels)}
labels = np.array([label_to_int[label] for label in labels_str])

# --- 5. Split Data ---
print("Splitting subset data into training and test sets...")
# A simple train/test split is sufficient for this smaller, balanced dataset
X_train, X_test, y_train, y_test = train_test_split(
    final_embeddings, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
)
print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# --- 6. Initialize Classifiers and Pipelines (Preserved from previous logic) ---
mnb_pipeline = Pipeline([
    ('mnb', MultinomialNB()) 
])
clf_dt = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=15)

# --- 7. Build FAISS Index and Define Prediction Function ---
print("Building FAISS index on training data...")
faiss_index = faiss.IndexFlatL2(X_train.shape[1])
faiss_index.add(X_train.astype(np.float32))
print("FAISS index built.")

def predict_with_faiss(data_to_predict):
    distances, indices = faiss_index.search(data_to_predict.astype(np.float32), KNN_N_NEIGHBORS)
    neighbor_labels = y_train[indices]
    predictions, _ = mode(neighbor_labels, axis=1)
    return predictions.ravel()

# --- 8. Fit, Predict, and Evaluate Ensembles ---
print("\nFitting individual models...")
mnb_pipeline.fit(X_train, y_train)
clf_dt.fit(X_train, y_train)
print("Models fitted.")

print("Generating predictions from all models on the test set...")
mnb_preds_test = mnb_pipeline.predict(X_test)
knn_preds_test = predict_with_faiss(X_test)
dt_preds_test = clf_dt.predict(X_test)
print("Predictions generated.")

# --- Evaluate Ensemble 1: MNB + kNN ---
print("\n--- Evaluating Ensemble 1: MNB + kNN (Manual Voting) on Test Set ---")
test_preds_eclf1 = np.stack([mnb_preds_test, knn_preds_test], axis=1)
combined_preds_eclf1, _ = mode(test_preds_eclf1, axis=1)
accuracy_eclf1_test = accuracy_score(y_test, combined_preds_eclf1)
print(f"Accuracy: {accuracy_eclf1_test:.4f}")
print("Classification Report:")
print(classification_report(y_test, combined_preds_eclf1, target_names=unique_labels, zero_division=0))

# --- Evaluate Ensemble 2: MNB + kNN + Decision Tree ---
print("\n--- Evaluating Ensemble 2: MNB + kNN + Decision Tree (Manual Voting) on Test Set ---")
test_preds_eclf2 = np.stack([mnb_preds_test, knn_preds_test, dt_preds_test], axis=1)
combined_preds_eclf2, _ = mode(test_preds_eclf2, axis=1)
accuracy_eclf2_test = accuracy_score(y_test, combined_preds_eclf2)
print(f"Accuracy: {accuracy_eclf2_test:.4f}")
print("Classification Report:")
print(classification_report(y_test, combined_preds_eclf2, target_names=unique_labels, zero_division=0))

print("\nEnsemble classification on subset complete.")