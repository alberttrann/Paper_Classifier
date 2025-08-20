# prepare_data_and_faiss.py

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
from tqdm.auto import tqdm
import torch # For device check
from collections import Counter # NEW: For counting class frequencies

# --- Configuration ---
DATASET_NAME = "UniverseTBD/arxiv-abstracts-large"
MODEL_NAME = "intfloat/multilingual-e5-base" # SBERT model for embeddings
OUTPUT_DIR = "processed_data"

# Total number of samples in the dataset
TOTAL_DATASET_SPLIT = "train"

# Chunk size for embedding generation. 458,000 samples is a good choice for 16GB RAM
# (2.29M / 458k = 5 chunks)
CHUNK_SIZE = 458000

TEST_SIZE = 0.15 # Proportion of data for the final test set
VALIDATION_SIZE_RATIO = 0.15 # Proportion of (remaining) data for the validation set
RANDOM_STATE = 42

# For RTX 4050 (6GB VRAM), BATCH_SIZE=64 is a good starting point.
# If you get CUDA out of memory errors, reduce to 32 or even 16.
BATCH_SIZE = 128

# Use 'cuda' to leverage your RTX 4050. Ensure PyTorch is installed with CUDA support.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    nltk.download('omw-1.4') # Often needed for WordNetLemmatizer

# --- Text Preprocessing Functions (Based on Teacher's Steps) ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Lowercasing
    text = text.lower()
    # Remove URLs and HTML tags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    # Handle emojis/emoticons (simple removal for now, could convert to words)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    # Remove punctuation and numbers
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text) # Remove numbers
    # Tokenization, Stop Word Removal, Lemmatization
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(cleaned_tokens)

print(f"Using device: {DEVICE}")

# --- 1. Load Dataset (Streaming) ---
print(f"Loading dataset: {DATASET_NAME} split {TOTAL_DATASET_SPLIT} in streaming mode...")
try:
    dataset_generator = load_dataset(DATASET_NAME, split=TOTAL_DATASET_SPLIT, streaming=True)
    print(f"Dataset loaded in streaming mode. Will process in chunks of {CHUNK_SIZE} samples.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure you have internet access and sufficient disk space.")
    print("Exiting. Cannot proceed without dataset.")
    exit()

# --- 2. Initialize Sentence Transformer Model ---
print(f"Loading SentenceTransformer model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)
model.eval() # Set model to evaluation mode

# --- 3. Process Data in Chunks and Generate Embeddings ---
all_labels_raw = []
temp_chunk_files = [] # To store paths to temporary numpy files

os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output directory exists

print(f"Starting chunked processing and embedding generation...")
current_chunk_abstracts = []
current_chunk_labels = []
chunk_count = 0

# Use enumerate with tqdm for progress over the stream
# Note: The total for tqdm will be unknown in streaming mode, but it still shows progress for current operations.
for item in tqdm(dataset_generator, desc="Processing Data"):
    abstract = item['abstract']
    category = item['categories']

    if abstract is None or category is None:
        continue # Skip corrupted entries

    # Clean text and append to current chunk
    cleaned_abstract = clean_text(abstract)
    current_chunk_abstracts.append(cleaned_abstract)
    current_chunk_labels.append(category)

    if len(current_chunk_abstracts) >= CHUNK_SIZE:
        chunk_count += 1
        print(f"\nGenerating embeddings for chunk {chunk_count} (size: {len(current_chunk_abstracts)})...")

        # Generate embeddings for the current chunk
        embeddings_chunk = model.encode(current_chunk_abstracts,
                                        batch_size=BATCH_SIZE,
                                        show_progress_bar=True,
                                        convert_to_numpy=True,
                                        device=DEVICE)

        # L2 Normalize Embeddings for this chunk
        embeddings_normalized_chunk = embeddings_chunk / np.linalg.norm(embeddings_chunk, axis=1, keepdims=True)

        # Save embeddings chunk to a temporary file
        chunk_file_path = os.path.join(OUTPUT_DIR, f'embeddings_chunk_{chunk_count}.npy')
        np.save(chunk_file_path, embeddings_normalized_chunk)
        temp_chunk_files.append(chunk_file_path)

        # Append labels for this chunk (as raw strings for now)
        all_labels_raw.extend(current_chunk_labels)

        # Clear current chunk data to free memory
        current_chunk_abstracts = []
        current_chunk_labels = []
        del embeddings_chunk # Explicitly delete to free GPU/CPU memory from raw embeddings
        del embeddings_normalized_chunk # Explicitly delete
        torch.cuda.empty_cache() # Clear GPU cache if using CUDA

# Process any remaining items in the last chunk
if len(current_chunk_abstracts) > 0:
    chunk_count += 1
    print(f"\nGenerating embeddings for final chunk {chunk_count} (size: {len(current_chunk_abstracts)})...")
    embeddings_chunk = model.encode(current_chunk_abstracts,
                                    batch_size=BATCH_SIZE,
                                    show_progress_bar=True,
                                    convert_to_numpy=True,
                                    device=DEVICE)
    embeddings_normalized_chunk = embeddings_chunk / np.linalg.norm(embeddings_chunk, axis=1, keepdims=True)
    chunk_file_path = os.path.join(OUTPUT_DIR, f'embeddings_chunk_{chunk_count}.npy')
    np.save(chunk_file_path, embeddings_normalized_chunk)
    temp_chunk_files.append(chunk_file_path)
    all_labels_raw.extend(current_chunk_labels)
    del embeddings_chunk
    del embeddings_normalized_chunk
    torch.cuda.empty_cache()

print(f"Finished generating embeddings for a total of {len(all_labels_raw)} samples across {chunk_count} chunks.")

# --- 4. Concatenate all Embeddings from Temporary Files ---
print("Concatenating all L2-normalized embedding chunks from disk...")
full_embeddings_list = []
for fpath in tqdm(temp_chunk_files, desc="Loading and concatenating chunks"):
    full_embeddings_list.append(np.load(fpath))
full_embeddings = np.vstack(full_embeddings_list)
print(f"Concatenated embeddings shape: {full_embeddings.shape}")

# Clean up temporary chunk files
for fpath in temp_chunk_files:
    os.remove(fpath)
print("Temporary chunk files removed.")

# --- 5. Map String Labels to Integers ---
print("Mapping string labels to integers...")
unique_labels_all = sorted(list(set(all_labels_raw)))
label_to_int_all = {label: i for i, label in enumerate(unique_labels_all)}
int_labels_full = np.array([label_to_int_all[label] for label in tqdm(all_labels_raw, desc="Mapping Labels")])
print(f"Total samples with embeddings and integer labels: {len(int_labels_full)}")

# --- NEW: Handle Singleton Classes (Classes with only 1 member) ---
print("Checking for and handling singleton classes...")
label_counts = Counter(int_labels_full)
# Identify labels that appear more than once
valid_label_indices = [label for label, count in label_counts.items() if count >= 2]

# Create a mask to filter out samples whose labels are singletons
mask = np.isin(int_labels_full, valid_label_indices)

# Apply mask to filter embeddings and labels
filtered_embeddings = full_embeddings[mask]
filtered_int_labels = int_labels_full[mask]

print(f"Original samples: {len(full_embeddings)}. Samples after filtering singletons: {len(filtered_embeddings)}")
if len(filtered_embeddings) != len(full_embeddings):
    print(f"Removed {len(full_embeddings) - len(filtered_embeddings)} samples belonging to singleton classes.")
    # Re-map filtered_int_labels to new contiguous integers if needed for classifiers,
    # but sklearn handles non-contiguous labels reasonably well for `stratify`.
    # Let's ensure the unique_labels list for saving is also updated.
    unique_labels = sorted([label_name for label_name, label_idx in label_to_int_all.items() if label_idx in valid_label_indices])
    # Re-map filtered labels to new contiguous indices for best practice
    new_label_to_int = {label: i for i, label in enumerate(unique_labels)}
    filtered_int_labels = np.array([new_label_to_int[unique_labels_all[old_idx]] for old_idx in filtered_int_labels])
else:
    unique_labels = unique_labels_all

# --- 6. Split Data ---
print("Splitting data into training, validation, and test sets...")
# First, split into training+validation and test
# Use the filtered data for splitting
X_train_val, X_test, y_train_val, y_test = train_test_split(
    filtered_embeddings, filtered_int_labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=filtered_int_labels
)

# Then, split training+validation into training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=VALIDATION_SIZE_RATIO / (1 - TEST_SIZE),
    random_state=RANDOM_STATE, stratify=y_train_val
)

print(f"Train set shape: {X_train.shape}, Labels: {y_train.shape}")
print(f"Validation set shape: {X_val.shape}, Labels: {y_val.shape}")
print(f"Test set shape: {X_test.shape}, Labels: {y_test.shape}")

# --- 7. Build FAISS Index for Training Embeddings (using IndexFlatIP for Cosine Similarity) ---
print("Building FAISS IndexFlatIP index for training embeddings...")
embedding_dim = filtered_embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(embedding_dim)
faiss_index.add(X_train) # Add the L2-normalized training embeddings to the index

print(f"FAISS index built. Number of vectors in index: {faiss_index.ntotal}")

# --- 8. Save Processed Data and FAISS Index ---
print(f"Saving processed data to '{OUTPUT_DIR}'...")
np.save(os.path.join(OUTPUT_DIR, 'X_train_emb.npy'), X_train)
np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(OUTPUT_DIR, 'X_val_emb.npy'), X_val)
np.save(os.path.join(OUTPUT_DIR, 'y_val.npy'), y_val)
np.save(os.path.join(OUTPUT_DIR, 'X_test_emb.npy'), X_test)
np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test)
joblib.dump(unique_labels, os.path.join(OUTPUT_DIR, 'unique_labels.pkl')) # Save the updated unique_labels
faiss.write_index(faiss_index, os.path.join(OUTPUT_DIR, 'faiss_index.bin'))

print("Data preparation and FAISS index creation complete.")