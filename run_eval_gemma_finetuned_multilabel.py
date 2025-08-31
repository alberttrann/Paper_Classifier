# evaluate_gemma_locally.py

import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from collections import Counter

# Unsloth is needed for loading the model and tokenizer correctly
from unsloth import FastLanguageModel
import torch

# NEW: Import PeftModel for correct adapter merging
from peft import PeftModel

# Scikit-learn for metrics and data splitting
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score, hamming_loss

from tqdm.auto import tqdm

# --- Configuration ---
# Set this to True to run the one-time merge and save process
PERFORM_MANUAL_MERGE = False 

# --- Model Paths ---
BASE_MODEL_NAME = "unsloth/gemma-3-270m-it" 
LORA_ADAPTER_PATH = "./gemma_classifier_lora"
CORRECT_MERGED_MODEL_PATH = "./gemma_classifier_merged_final"

# Data Sampling Configuration (MUST MATCH YOUR TRAINING RUN)
CATEGORIES_TO_SELECT = [
    'math', 'astro-ph', 'cs', 'cond-mat', 'physics',
    'hep-ph', 'quant-ph', 'hep-th'
]
SAMPLES_PER_CATEGORY_APPEARANCE = 2500
RANDOM_STATE = 42
max_seq_length = 2048

# --- Main Execution ---

# ==============================================================================
# PART 1: MANUAL MERGE AND SAVE (RUN THIS ONCE)
# ==============================================================================
if PERFORM_MANUAL_MERGE:
    print(f"--- Starting Manual Merge & Save Process ---")
    
    # --- THIS IS THE CORRECTED MERGE LOGIC ---
    print(f"Loading base model '{BASE_MODEL_NAME}' for merging...")
    
    # Load the base model in 4bit for memory efficiency during the initial load
    # The final merged model will be in a higher precision.
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = BASE_MODEL_NAME,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )
    
    print(f"Applying LoRA adapters from '{LORA_ADAPTER_PATH}'...")
    # Use PeftModel to apply the adapters to the base model
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    
    print("Merging adapters into the model...")
    # The merge_and_unload() function correctly combines the weights
    # and returns a standard Hugging Face model.
    model = model.merge_and_unload()
    
    print(f"Saving the fully merged, standalone model to '{CORRECT_MERGED_MODEL_PATH}'...")
    # Now we can save the new, merged model
    model.save_pretrained(CORRECT_MERGED_MODEL_PATH)
    tokenizer.save_pretrained(CORRECT_MERGED_MODEL_PATH)
    
    print("\n--- Manual Merge & Save Complete ---")
    print("You can now set PERFORM_MANUAL_MERGE to False.")
    exit()
    # --- END OF CORRECTED MERGE LOGIC ---


# ==============================================================================
# PART 2: EVALUATION (The main part of the script)
# ==============================================================================

# 1. Data Sampling and Test Set Preparation
print("--- Step 1: Recreating the Exact Test Set ---")
category_counts = {cat: 0 for cat in CATEGORIES_TO_SELECT}
samples = []
dataset_generator = load_dataset("UniverseTBD/arxiv-abstracts-large", split="train", streaming=True)
for s in tqdm(dataset_generator, desc="Scanning for samples"):
    if all(count >= SAMPLES_PER_CATEGORY_APPEARANCE for count in category_counts.values()):
        break
    if s['categories'] is None or s['abstract'] is None: continue
    parent_categories = {cat.split('.')[0] for cat in s['categories'].strip().split(' ')}
    if any(p in CATEGORIES_TO_SELECT for p in parent_categories):
        relevant_categories = [p for p in parent_categories if p in CATEGORIES_TO_SELECT]
        if relevant_categories:
            samples.append({'abstract': s['abstract'], 'categories': sorted(relevant_categories)})
            for p_cat in relevant_categories:
                category_counts[p_cat] += 1
print(f"Finished sampling. Total samples collected: {len(samples)}")
df_samples = pd.DataFrame(samples)
_, test_df = train_test_split(df_samples, test_size=0.1, random_state=RANDOM_STATE)
print(f"Isolated {len(test_df)} samples for the test set.")

# 2. Load the Fine-Tuned Model and Tokenizer (using LoRA adapters)
print(f"\n--- Step 2: Loading model with LoRA adapters from '{LORA_ADAPTER_PATH}' for evaluation ---")
# Unsloth's from_pretrained for adapters is the most reliable way to load for inference
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = LORA_ADAPTER_PATH,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)
print("Model and tokenizer loaded successfully.")

# 3. Evaluation
print("\n--- Step 3: Starting Evaluation on Test Set ---")
# ... (The evaluation logic is exactly the same as before and remains correct) ...
true_labels_str = [sorted(x) for x in test_df['categories']]
mlb = MultiLabelBinarizer(classes=CATEGORIES_TO_SELECT)
Y_true = mlb.fit_transform(true_labels_str)
category_list_str = ", ".join(CATEGORIES_TO_SELECT)
eval_prompt_template = """<start_of_turn>user
Classify the following scientific abstract into one or more of the predefined categories.

Categories: {}

Abstract:
{}<end_of_turn>
<start_of_turn>model"""
predicted_labels_str = []
for index, example in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating predictions"):
    prompt = eval_prompt_template.format(category_list_str, example['abstract'])
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id, do_sample=False)
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    answer = decoded_output.split("<start_of_turn>model")[-1].strip()
    parsed_labels = [label.strip() for label in answer.split(',')]
    valid_labels = sorted([label for label in parsed_labels if label in CATEGORIES_TO_SELECT])
    predicted_labels_str.append(valid_labels)

Y_pred = mlb.transform(predicted_labels_str)
subset_acc = accuracy_score(Y_true, Y_pred)
hamming = hamming_loss(Y_true, Y_pred)
report = classification_report(Y_true, Y_pred, target_names=CATEGORIES_TO_SELECT, zero_division=0)

print("\n" + "="*50)
print(f"--- Gemma Fine-Tuning Results (LoRA model) ---")
print("="*50)
print(f"Overall Subset Accuracy: {subset_acc:.4f}")
print(f"Hamming Loss: {hamming:.4f}\n")
print("Per-Category Performance:")
print(report)