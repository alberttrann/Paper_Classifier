# run_ensemble_classification.py

import os
import numpy as np
import joblib
import faiss
import time # Import time to measure duration
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from scipy.stats import mode
from collections import Counter

# --- SCRIPT START ---
print("--- SCRIPT EXECUTION STARTED ---")
start_time = time.time()

# --- Configuration ---
OUTPUT_DIR = "processed_data"
KNN_N_NEIGHBORS = 5
RANDOM_STATE = 42
TOP_K_CLASSES = 1000

# --- Load Processed Data ---
print(f"Loading final processed data from '{OUTPUT_DIR}'...")
try:
    X_train = np.load(os.path.join(OUTPUT_DIR, 'X_train_emb.npy'))
    y_train = np.load(os.path.join(OUTPUT_DIR, 'y_train.npy'))
    X_val = np.load(os.path.join(OUTPUT_DIR, 'X_val_emb.npy'))
    y_val = np.load(os.path.join(OUTPUT_DIR, 'y_val.npy'))
    X_test = np.load(os.path.join(OUTPUT_DIR, 'X_test_emb.npy'))
    y_test = np.load(os.path.join(OUTPUT_DIR, 'y_test.npy'))
    original_unique_labels = joblib.load(os.path.join(OUTPUT_DIR, 'unique_labels.pkl'))
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: Could not find data file '{e.filename}'.")
    print("Please ensure your data from the 6-hour prep run exists and filenames match.")
    exit()

# --- Prune the number of classes ---
print(f"Dataset has {len(original_unique_labels)} unique classes. Pruning to the top {TOP_K_CLASSES}...")
label_counts = Counter(y_train)
top_k_label_indices = {label for label, count in label_counts.most_common(TOP_K_CLASSES)}
other_label_id = TOP_K_CLASSES
label_map = np.full(max(label_counts.keys()) + 1, other_label_id, dtype=np.int32)
for i, label_idx in enumerate(top_k_label_indices):
    label_map[label_idx] = i
y_train_pruned = label_map[y_train]
y_val_pruned = label_map[y_val]
y_test_pruned = label_map[y_test]
top_k_original_indices = [item[0] for item in label_counts.most_common(TOP_K_CLASSES)]
target_names = [original_unique_labels[i] for i in top_k_original_indices] + ['other']
final_labels_for_report = np.arange(len(target_names))
print(f"Labels pruned. New number of classes: {len(target_names)}")


# --- Initialize Classifiers and Pipelines ---
mnb_pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('mnb', MultinomialNB())
])
clf_dt = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=15)

# --- Build FAISS Index ---
print("Building FAISS index on training data...")
faiss_index = faiss.IndexFlatL2(X_train.shape[1])
faiss_index.add(X_train.astype(np.float32))
print("FAISS index built.")

# --- Helper function for FAISS-accelerated prediction ---
def predict_with_faiss(data_to_predict):
    distances, indices = faiss_index.search(data_to_predict.astype(np.float32), KNN_N_NEIGHBORS)
    neighbor_labels_original = y_train[indices]
    neighbor_labels_pruned = label_map[neighbor_labels_original]
    predictions, _ = mode(neighbor_labels_pruned, axis=1)
    return predictions.ravel()

# --- Fit the individual models ---
print("\nFitting individual models (This may take some time due to RAM pressure)...")
fit_start_time = time.time()
mnb_pipeline.fit(X_train, y_train_pruned)
clf_dt.fit(X_train, y_train_pruned)
fit_end_time = time.time()
print(f"Models fitted in {fit_end_time - fit_start_time:.2f} seconds.")

# --- Generate predictions from all models ---
print("Generating predictions from all models on validation and test sets...")
mnb_preds_val = mnb_pipeline.predict(X_val)
knn_preds_val = predict_with_faiss(X_val)
dt_preds_val = clf_dt.predict(X_val)

mnb_preds_test = mnb_pipeline.predict(X_test)
knn_preds_test = predict_with_faiss(X_test)
dt_preds_test = clf_dt.predict(X_test)
print("Predictions generated.")

# --- Evaluate Ensemble 1 on Validation Set ---
print("\n--- Evaluating Ensemble 1: MNB + kNN (Manual Voting) on Validation Set ---")
val_preds_eclf1 = np.stack([mnb_preds_val, knn_preds_val], axis=1)
combined_preds_eclf1, _ = mode(val_preds_eclf1, axis=1)
accuracy_eclf1_val = accuracy_score(y_val_pruned, combined_preds_eclf1)
print(f"Accuracy: {accuracy_eclf1_val:.4f}")
print("Classification Report:")
print(classification_report(y_val_pruned, combined_preds_eclf1, labels=final_labels_for_report, target_names=target_names, zero_division=0))

# --- Evaluate Ensemble 2 on Validation Set ---
print("\n--- Evaluating Ensemble 2: MNB + kNN + Decision Tree (Manual Voting) on Validation Set ---")
val_preds_eclf2 = np.stack([mnb_preds_val, knn_preds_val, dt_preds_val], axis=1)
combined_preds_eclf2, _ = mode(val_preds_eclf2, axis=1)
accuracy_eclf2_val = accuracy_score(y_val_pruned, combined_preds_eclf2)
print(f"Accuracy: {accuracy_eclf2_val:.4f}")
print("Classification Report:")
print(classification_report(y_val_pruned, combined_preds_eclf2, labels=final_labels_for_report, target_names=target_names, zero_division=0))

# --- Final Comparison and Test Set Evaluation ---
print("\n--- Final Comparison and Test Set Evaluation ---")
best_ensemble_val_accuracy = max(accuracy_eclf1_val, accuracy_eclf2_val)
best_ensemble_name = "MNB + kNN" if accuracy_eclf1_val >= accuracy_eclf2_val else "MNB + kNN + DT"

print(f"\nBest performing ensemble on Validation Set: {best_ensemble_name} with Accuracy: {best_ensemble_val_accuracy:.4f}")
print(f"Evaluating the best ensemble on the Test Set:")

if best_ensemble_name == "MNB + kNN":
    test_preds = np.stack([mnb_preds_test, knn_preds_test], axis=1)
else: # MNB + kNN + DT
    test_preds = np.stack([mnb_preds_test, knn_preds_test, dt_preds_test], axis=1)
final_test_preds, _ = mode(test_preds, axis=1)
accuracy_test = accuracy_score(y_test_pruned, final_test_preds)
print(f"Final Test Set Accuracy: {accuracy_test:.4f}")
print("Final Test Set Classification Report:")
print(classification_report(y_test_pruned, final_test_preds, labels=final_labels_for_report, target_names=target_names, zero_division=0))

end_time = time.time()
print(f"\n--- SCRIPT EXECUTION FINISHED in {(end_time - start_time) / 60:.2f} minutes ---")