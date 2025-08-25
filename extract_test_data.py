# prepare_final_datasets.py

import os
import re
import string
import numpy as np
import pandas as pd
from datasets import load_dataset
from collections import Counter

# Scikit-learn for the train/test split
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# --- Configuration ---
RANDOM_STATE = 42
OUTPUT_DIR = "final_datasets"

# Single-label configuration
SINGLE_LABEL_CATEGORIES = ['math', 'astro-ph', 'cs', 'cond-mat', 'physics', 'hep-ph', 'quant-ph', 'hep-th']
SAMPLES_PER_SINGLE_LABEL_CAT = 5000

# Multi-label configuration
MULTI_LABEL_CATEGORIES = ['math', 'astro-ph', 'cs', 'cond-mat', 'physics', 'hep-ph', 'quant-ph', 'hep-th']
SAMPLES_PER_MULTI_LABEL_APPEARANCE = 5000

# --- Main Execution ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Part 1: Prepare SINGLE-LABEL Train/Test Sets ===
print("--- Starting SINGLE-LABEL data sampling and splitting ---")
category_counts_sl = {cat: 0 for cat in SINGLE_LABEL_CATEGORIES}
samples_sl = []
# Use a fresh generator instance
dataset_generator_sl = load_dataset("UniverseTBD/arxiv-abstracts-large", split="train", streaming=True)

for s in tqdm(dataset_generator_sl, desc="Scanning for single-label samples"):
    # Stop when all quotas are met
    if all(count >= SAMPLES_PER_SINGLE_LABEL_CAT for count in category_counts_sl.values()):
        break
    # Check for valid, single-label entries
    if s['categories'] is None or s['abstract'] is None or len(s['categories'].split(' ')) != 1:
        continue
    parent_category = s['categories'].strip().split('.')[0]
    if parent_category in SINGLE_LABEL_CATEGORIES and category_counts_sl[parent_category] < SAMPLES_PER_SINGLE_LABEL_CAT:
        samples_sl.append({'abstract': s['abstract'], 'category': parent_category})
        category_counts_sl[parent_category] += 1

print(f"Collected {len(samples_sl)} single-label samples.")
abstracts_sl = [s['abstract'] for s in samples_sl]
labels_sl_str = [s['category'] for s in samples_sl]

# Perform the 80/20 split
train_abstracts_sl, test_abstracts_sl, train_labels_sl, test_labels_sl = train_test_split(
    abstracts_sl, labels_sl_str, test_size=0.2, random_state=RANDOM_STATE, stratify=labels_sl_str
)

# Save both train and test sets to CSV
df_sl_train = pd.DataFrame({'abstract': train_abstracts_sl, 'category': train_labels_sl})
df_sl_test = pd.DataFrame({'abstract': test_abstracts_sl, 'category': test_labels_sl})
df_sl_train.to_csv(os.path.join(OUTPUT_DIR, 'single_label_train.csv'), index=False)
df_sl_test.to_csv(os.path.join(OUTPUT_DIR, 'single_label_test.csv'), index=False)
print(f"Saved {len(df_sl_train)} train and {len(df_sl_test)} test samples for single-label task.")


# === Part 2: Prepare MULTI-LABEL Train/Test Sets ===
print("\n--- Starting MULTI-LABEL data sampling and splitting ---")
category_counts_ml = {cat: 0 for cat in MULTI_LABEL_CATEGORIES}
samples_ml = []
# Use a fresh generator instance
dataset_generator_ml = load_dataset("UniverseTBD/arxiv-abstracts-large", split="train", streaming=True)

for s in tqdm(dataset_generator_ml, desc="Scanning for multi-label samples"):
    # Stop when all quotas are met
    if all(count >= SAMPLES_PER_MULTI_LABEL_APPEARANCE for count in category_counts_ml.values()):
        break
    if s['categories'] is None or s['abstract'] is None:
        continue
    parent_categories = {cat.split('.')[0] for cat in s['categories'].strip().split(' ')}
    # This is the exact logic from the deep learning script
    if any(p in MULTI_LABEL_CATEGORIES for p in parent_categories):
        samples_ml.append({'abstract': s['abstract'], 'categories': list(parent_categories)})
        for p_cat in parent_categories:
            if p_cat in category_counts_ml:
                category_counts_ml[p_cat] += 1

print(f"Collected {len(samples_ml)} multi-label samples.")
abstracts_ml = [s['abstract'] for s in samples_ml]
labels_ml = [s['categories'] for s in samples_ml]

# Perform the 80/20 split (without stratification for robustness)
train_abstracts_ml, test_abstracts_ml, train_labels_ml, test_labels_ml = train_test_split(
    abstracts_ml, labels_ml, test_size=0.2, random_state=RANDOM_STATE
)

# Save both train and test sets to CSV
df_ml_train = pd.DataFrame({'abstract': train_abstracts_ml, 'categories': train_labels_ml})
df_ml_test = pd.DataFrame({'abstract': test_abstracts_ml, 'categories': test_labels_ml})
df_ml_train.to_csv(os.path.join(OUTPUT_DIR, 'multi_label_train.csv'), index=False)
df_ml_test.to_csv(os.path.join(OUTPUT_DIR, 'multi_label_test.csv'), index=False)
print(f"Saved {len(df_ml_train)} train and {len(df_ml_test)} test samples for multi-label task.")

print(f"\nData preparation complete. All datasets saved in '{OUTPUT_DIR}'.")