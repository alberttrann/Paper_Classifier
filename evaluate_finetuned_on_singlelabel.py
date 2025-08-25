# evaluate_finetuned_on_singlelabel.py

import os
import re
import string
import numpy as np
import pandas as pd
from datetime import datetime

# NLTK for text cleaning
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# PyTorch and Transformers
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

# Scikit-learn for metrics
from sklearn.metrics import accuracy_score, classification_report

from tqdm.auto import tqdm

# --- Configuration ---
# Path to your saved fine-tuned model
FINETUNED_MODEL_PATH = "./e5_finetuned_multilabel" 
# Path to the single-label test data
TEST_DATA_PATH = "./final_datasets/single_label_test.csv"

# These must match the categories the model was trained on
CATEGORIES = [
    'math', 'astro-ph', 'cs', 'cond-mat', 'physics', 
    'hep-ph', 'quant-ph', 'hep-th'
]

BATCH_SIZE = 32 # Can be larger for inference
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- NLTK Downloads & Text Preprocessing (Consistent with training) ---
try:
    stopwords.words('english')
except LookupError:
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

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

# --- PyTorch Dataset Class (for evaluation) ---
class ArxivSingleLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label_map, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item_idx):
        text = self.texts[item_idx]
        label_str = self.labels[item_idx]
        label_id = self.label_map[label_str]
        
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
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label_id, dtype=torch.long)
        }

# --- Custom Transformer Model Definition (Must match the one used for training) ---
class MultiLabelTransformer(torch.nn.Module):
    def __init__(self, base_model_path, n_classes):
        super(MultiLabelTransformer, self).__init__()
        self.transformer = AutoModel.from_pretrained(base_model_path)
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(self.transformer.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        transformer_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = transformer_output.pooler_output
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        return logits

# --- Evaluation Function ---
def evaluate_on_single_label(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating on Single-Label Data"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            
            # --- KEY LOGIC CHANGE ---
            # Instead of thresholding, we find the single highest probability
            # The output of argmax is the index of the predicted class
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)


# --- Main Execution ---
print(f"--- Evaluating Fine-Tuned e5 Model on Single-Label Test Set ---")
print(f"Using device: {DEVICE}")

# 1. Load the fine-tuned model and tokenizer
print(f"Loading model from: {FINETUNED_MODEL_PATH}")
try:
    # Recreate the model structure
    model = MultiLabelTransformer(FINETUNED_MODEL_PATH, n_classes=len(CATEGORIES))
    # Load the saved classifier weights
    classifier_weights_path = os.path.join(FINETUNED_MODEL_PATH, "classifier_weights.bin")
    model.classifier.load_state_dict(torch.load(classifier_weights_path, map_location=torch.device(DEVICE)))
    model = model.to(DEVICE)
    
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
except FileNotFoundError:
    print(f"ERROR: Model or tokenizer not found at '{FINETUNED_MODEL_PATH}'.")
    print("Please ensure you have downloaded and placed the 'e5_finetuned_multilabel' folder correctly.")
    exit()
print("Model and tokenizer loaded successfully.")

# 2. Load and prepare the single-label test data
print(f"Loading single-label test data from: {TEST_DATA_PATH}")
try:
    df_test = pd.read_csv(TEST_DATA_PATH)
except FileNotFoundError:
    print(f"ERROR: Test data not found at '{TEST_DATA_PATH}'.")
    print("Please run 'prepare_final_datasets.py' first.")
    exit()

print("Cleaning and preparing test data...")
df_test['cleaned_abstract'] = [clean_text(abstract) for abstract in tqdm(df_test['abstract'], desc="Cleaning test abstracts")]

# Create label map and dataset
cat_to_idx = {cat: i for i, cat in enumerate(CATEGORIES)}
test_dataset = ArxivSingleLabelDataset(
    texts=df_test['cleaned_abstract'].tolist(),
    labels=df_test['category'].tolist(),
    tokenizer=tokenizer,
    label_map=cat_to_idx
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 3. Run Evaluation
print("\n--- Starting Evaluation ---")
y_pred, y_true = evaluate_on_single_label(model, test_loader, DEVICE)

# 4. Display and Log Results
accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=CATEGORIES)

report_header = "\n\n" + "="*80
report_header += f"\n--- Evaluation: Fine-Tuned e5 on Single-Label Task ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---"
report_header += "\n" + "="*80

report_body = f"\nOverall Single-Label Accuracy: {accuracy:.4f}\n\n"
report_body += "Per-Category Performance:\n"
report_body += report

# Utility function to log messages to console and/or file
def log_message(message, to_console=True, log_file_path=None):
    if to_console:
        print(message)
    if log_file_path is not None:
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

# Log to the main benchmark file
LOG_FILE_PATH = os.path("multilabel_finetuned_on_singlelabel.txt")
log_message(report_header, to_console=True, log_file_path=LOG_FILE_PATH)
log_message(report_body, to_console=True, log_file_path=LOG_FILE_PATH)

print(f"\nEvaluation complete. Results appended to '{LOG_FILE_PATH}'.")