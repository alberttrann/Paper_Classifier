# run_deep_learning_multilabel.py

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

# --- NEW: PyTorch and Transformers Imports ---
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW

# Scikit-learn for metrics and data splitting
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, hamming_loss

from tqdm.auto import tqdm

# --- Configuration ---
# Data Sampling
CATEGORIES_TO_SELECT = [
    'math', 'astro-ph', 'cs', 'cond-mat', 'physics',
    'hep-ph', 'quant-ph', 'hep-th'
]
SAMPLES_PER_CATEGORY_APPEARANCE = 5000

# Model & Training
E5_MODEL_NAME = "intfloat/multilingual-e5-base"
RANDOM_STATE = 42
BATCH_SIZE = 8 # Reduced batch size to mitigate OOM error
EPOCHS = 4
LEARNING_RATE = 2e-5 # A standard learning rate for fine-tuning
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_FILE_PATH = "deep_learning_multilabel.txt"

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
try:
    import nltk
    nltk.data.find('tokenizers/punkt_tab/english/')
except LookupError:
    import nltk
    nltk.download('punkt_tab')


# --- Helper function for logging ---
def log_message(message, to_console=True):
    if to_console:
        print(message)
    with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

# --- Enhanced Text Preprocessing Function ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
domain_specific_stopwords = {'result', 'study', 'show', 'paper', 'model', 'analysis', 'method', 'approach', 'propose', 'demonstrate', 'investigate'}
stop_words.update(domain_specific_stopwords)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(cleaned_tokens)

# --- NEW: PyTorch Dataset Class ---
class ArxivMultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item_idx):
        text = self.texts[item_idx]
        label = self.labels[item_idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }

# --- NEW: Custom Transformer Model for Multi-Label Classification ---
class MultiLabelTransformer(torch.nn.Module):
    def __init__(self, base_model_name, n_classes):
        super(MultiLabelTransformer, self).__init__()
        self.transformer = AutoModel.from_pretrained(base_model_name)
        # Add a dropout layer for regularization
        self.dropout = torch.nn.Dropout(0.2)
        # The final linear layer that maps the embedding to our 8 classes
        self.classifier = torch.nn.Linear(self.transformer.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        # Get the embeddings from the base transformer
        transformer_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use the embedding of the [CLS] token for classification
        pooled_output = transformer_output.pooler_output

        # Apply dropout and the final classification layer
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        return logits

# --- NEW: Training and Evaluation Functions ---
def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)

        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(data_loader)

def eval_model(model, data_loader, device, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            # Apply sigmoid to get probabilities, then apply threshold
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)

# --- Main Execution ---
log_message("\n\n" + "="*80)
log_message(f"--- Deep Learning Multi-Label Benchmark: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
log_message("="*80)

# 1. Multi-Label Data Sampling and Preparation
print("--- Step 1: Multi-Label Data Sampling & Preparation ---")
category_counts = {cat: 0 for cat in CATEGORIES_TO_SELECT}
samples = []
dataset_generator = load_dataset("UniverseTBD/arxiv-abstracts-large", split="train", streaming=True)
for s in tqdm(dataset_generator, desc="Scanning for samples"):
    if all(count >= SAMPLES_PER_CATEGORY_APPEARANCE for count in category_counts.values()):
        break
    if s['categories'] is None or s['abstract'] is None: continue
    parent_categories = {cat.split('.')[0] for cat in s['categories'].strip().split(' ')}
    if any(p in CATEGORIES_TO_SELECT for p in parent_categories):
        samples.append({'abstract': s['abstract'], 'parent_categories': parent_categories})
        for p_cat in parent_categories:
            if p_cat in category_counts:
                category_counts[p_cat] += 1
print(f"Finished sampling. Total samples collected: {len(samples)}")
abstracts = [sample['abstract'] for sample in samples]
labels_sets = [sample['parent_categories'] for sample in samples]
processed_abstracts = [clean_text(abstract) for abstract in tqdm(abstracts, desc="Cleaning Abstracts")]
Y = np.zeros((len(samples), len(CATEGORIES_TO_SELECT)), dtype=int)
cat_to_idx = {cat: i for i, cat in enumerate(CATEGORIES_TO_SELECT)}
for i, label_set in enumerate(labels_sets):
    for label in label_set:
        if label in cat_to_idx:
            Y[i, cat_to_idx[label]] = 1

train_texts, test_texts, Y_train, Y_test = train_test_split(
    processed_abstracts, Y, test_size=0.2, random_state=RANDOM_STATE
)

# 2. Tokenization and Dataset Creation
print("\n--- Step 2: Tokenizing Text and Creating PyTorch Datasets ---")
tokenizer = AutoTokenizer.from_pretrained(E5_MODEL_NAME)
train_dataset = ArxivMultiLabelDataset(train_texts, Y_train, tokenizer)
test_dataset = ArxivMultiLabelDataset(test_texts, Y_test, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 3. Model Initialization
print("\n--- Step 3: Initializing Model, Loss Function, and Optimizer ---")
model = MultiLabelTransformer(E5_MODEL_NAME, n_classes=len(CATEGORIES_TO_SELECT))
model = model.to(DEVICE)
# Use BCEWithLogitsLoss for multi-label classification
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# 4. Training Loop
print("\n--- Step 4: Starting Fine-Tuning Loop ---")
for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
    train_loss = train_epoch(model, train_loader, loss_fn, optimizer, DEVICE)
    print(f"  Train loss: {train_loss:.4f}")

# 5. Final Evaluation
print("\n--- Step 5: Final Evaluation on Test Set ---")
Y_pred, Y_true = eval_model(model, test_loader, DEVICE)

# Log results
accuracy = accuracy_score(Y_true, Y_pred)
hamming = hamming_loss(Y_true, Y_pred)
report = classification_report(Y_true, Y_pred, target_names=CATEGORIES_TO_SELECT, zero_division=0)

log_message("\n" + "="*50 + f"\nModel: Fine-Tuned Transformer ({E5_MODEL_NAME})\n" + "="*50)
log_message(f"Overall Subset Accuracy: {accuracy:.4f}")
log_message(f"Hamming Loss: {hamming:.4f}\n")
log_message("Per-Category Performance:")
log_message(report)

print(f"\nDeep learning benchmark complete. Results appended to '{LOG_FILE_PATH}'.")