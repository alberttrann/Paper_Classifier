# analyze_categories.py

from datasets import load_dataset
from collections import Counter
from tqdm.auto import tqdm

# --- Configuration ---
DATASET_NAME = "UniverseTBD/arxiv-abstracts-large"
DATASET_SPLIT = "train"

# --- Main Execution ---

print(f"Analyzing categories for dataset: {DATASET_NAME} (split: {DATASET_SPLIT})")
print("This may take a few minutes as it iterates through all 2.29M samples...")

# Load the dataset in streaming mode to avoid loading everything into RAM
try:
    dataset_generator = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True)
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure you have an internet connection.")
    exit()

# --- Initialize Counters ---
# Counter for samples with only one label (e.g., 'cs.LG')
single_label_primary_counts = Counter()

# Counter for all primary categories that appear, including in multi-label samples
all_primary_category_counts = Counter()

# --- Iterate Through the Dataset ---
# The total number of samples is known for this dataset, which helps tqdm
# If it were a truly unknown stream, we wouldn't set total.
total_samples = 2292057 

for sample in tqdm(dataset_generator, total=total_samples, desc="Scanning samples"):
    categories_str = sample.get('categories')
    
    if not categories_str:
        continue

    # Split the categories string into a list of individual labels
    # e.g., "cs.LG math.ST" -> ['cs.LG', 'math.ST']
    labels = categories_str.strip().split(' ')
    
    # --- Part 1: Count Single-Label Primary Categories ---
    # We only care about entries that have exactly one label
    if len(labels) == 1:
        single_label = labels[0]
        # Extract the primary category (e.g., 'cs' from 'cs.LG')
        primary_category = single_label.split('.')[0]
        single_label_primary_counts[primary_category] += 1
        
    # --- Part 2: Count ALL Unique Primary Categories ---
    # We process every entry, including multi-label ones
    unique_primary_categories_in_sample = set()
    for label in labels:
        primary_category = label.split('.')[0]
        unique_primary_categories_in_sample.add(primary_category)
    
    # Add the unique primary categories from this sample to the main counter
    all_primary_category_counts.update(unique_primary_categories_in_sample)

# --- Display Results ---

print("\n\n" + "="*80)
print("Analysis Complete. Here are the results:")
print("="*80)


# --- Print Results for Single-Label Entries ---
print("\n--- Counts for SINGLE-LABEL Primary Categories ---")
print(f"{'Primary Category':<25} | {'Number of Samples':<20}")
print("-" * 50)

# Sort the counter by the number of samples in descending order
for category, count in single_label_primary_counts.most_common():
    print(f"{category:<25} | {count:<20}")

print("-" * 50)
print(f"Found {len(single_label_primary_counts)} unique primary categories in single-label entries.")


# --- Print Results for All Entries ---
print("\n\n--- Counts for ALL Primary Categories (including multi-label entries) ---")
print("This shows the overall prevalence of each topic in the entire dataset.\n")
print(f"{'Primary Category':<25} | {'Total Appearances':<20}")
print("-" * 50)

# Sort the counter by the number of samples in descending order
for category, count in all_primary_category_counts.most_common():
    print(f"{category:<25} | {count:<20}")

print("-" * 50)
print(f"Found {len(all_primary_category_counts)} unique primary categories in total.")