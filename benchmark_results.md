# Benchmark Results

## 1. Configuration

- **Categories**: `['astro-ph', 'cond-mat', 'cs', 'math', 'physics']`
- **Dataset**: `UniverseTBD/arxiv-abstracts-large`
- **Device**: `cuda`
- **Embedding Model**: `intfloat/multilingual-e5-base`

---

## 2. Single Model Benchmarks (`run_single_benchmark.py`)

*   **Samples per category**: 1000

### 2.1. k-Nearest Neighbors (kNN)

<details>
<summary>kNN with Bag of Words</summary>

**Overall Accuracy**: 0.3500
```
              precision    recall  f1-score   support
           0       0.62      0.40      0.48       200
           1       0.57      0.08      0.14       200
           2       0.33      0.43      0.38       200
           3       0.27      0.73      0.40       200
           4       0.47      0.10      0.16       200
    accuracy                           0.35      1000
   macro avg       0.45      0.35      0.31      1000
weighted avg       0.45      0.35      0.31      1000
```
</details>

<details>
<summary>kNN with TF-IDF</summary>

**Overall Accuracy**: 0.8010
```
              precision    recall  f1-score   support
           0       0.87      0.90      0.88       200
           1       0.78      0.84      0.81       200
           2       0.83      0.80      0.81       200
           3       0.84      0.88      0.86       200
           4       0.68      0.58      0.63       200
    accuracy                           0.80      1000
   macro avg       0.80      0.80      0.80      1000
weighted avg       0.80      0.80      0.80      1000
```
</details>

<details>
<summary>kNN with Embeddings</summary>

**Overall Accuracy**: 0.8590
```
              precision    recall  f1-score   support
           0       0.92      0.95      0.94       200
           1       0.76      0.94      0.84       200
           2       0.90      0.91      0.90       200
           3       0.89      0.92      0.91       200
           4       0.85      0.58      0.69       200
    accuracy                           0.86      1000
   macro avg       0.86      0.86      0.85      1000
weighted avg       0.86      0.86      0.85      1000
```
</details>

### 2.2. Multinomial Naive Bayes (MNB)

<details>
<summary>MNB with Bag of Words</summary>

**Overall Accuracy**: 0.8710
```
              precision    recall  f1-score   support
           0       0.98      0.89      0.93       200
           1       0.84      0.93      0.88       200
           2       0.89      0.88      0.89       200
           3       0.91      0.97      0.94       200
           4       0.74      0.69      0.71       200
    accuracy                           0.87      1000
   macro avg       0.87      0.87      0.87      1000
weighted avg       0.87      0.87      0.87      1000
```
</details>

<details>
<summary>MNB with TF-IDF</summary>

**Overall Accuracy**: 0.8670
```
              precision    recall  f1-score   support
           0       0.94      0.93      0.93       200
           1       0.81      0.95      0.87       200
           2       0.88      0.88      0.88       200
           3       0.91      0.96      0.94       200
           4       0.79      0.61      0.69       200
    accuracy                           0.87      1000
   macro avg       0.87      0.87      0.86      1000
weighted avg       0.87      0.87      0.86      1000
```
</details>

<details>
<summary>MNB with Embeddings</summary>

**Overall Accuracy**: 0.8160
```
              precision    recall  f1-score   support
           0       0.90      0.94      0.92       200
           1       0.78      0.83      0.80       200
           2       0.83      0.84      0.84       200
           3       0.86      0.93      0.89       200
           4       0.67      0.54      0.60       200
    accuracy                           0.82      1000
   macro avg       0.81      0.82      0.81      1000
weighted avg       0.81      0.82      0.81      1000
```
</details>

### 2.3. Decision Tree (DT)

<details>
<summary>DT with Bag of Words</summary>

**Overall Accuracy**: 0.6130
```
              precision    recall  f1-score   support
           0       0.88      0.74      0.80       200
           1       0.67      0.62      0.64       200
           2       0.64      0.67      0.65       200
           3       0.75      0.54      0.62       200
           4       0.34      0.50      0.41       200
    accuracy                           0.61      1000
   macro avg       0.65      0.61      0.63      1000
weighted avg       0.65      0.61      0.63      1000
```
</details>

<details>
<summary>DT with TF-IDF</summary>

**Overall Accuracy**: 0.6200
```
              precision    recall  f1-score   support
           0       0.86      0.72      0.78       200
           1       0.69      0.54      0.60       200
           2       0.68      0.64      0.66       200
           3       0.77      0.57      0.66       200
           4       0.37      0.63      0.47       200
    accuracy                           0.62      1000
   macro avg       0.67      0.62      0.63      1000
weighted avg       0.67      0.62      0.63      1000
```
</details>

<details>
<summary>DT with Embeddings</summary>

**Overall Accuracy**: 0.5110
```
              precision    recall  f1-score   support
           0       0.64      0.65      0.64       200
           1       0.48      0.54      0.51       200
           2       0.54      0.49      0.51       200
           3       0.56      0.58      0.57       200
           4       0.33      0.30      0.31       200
    accuracy                           0.51      1000
   macro avg       0.51      0.51      0.51      1000
weighted avg       0.51      0.51      0.51      1000
```
</details>

### 2.4. K-Means Clustering

<details>
<summary>KMeans with Bag of Words</summary>

**Overall Accuracy**: 0.3880
```
              precision    recall  f1-score   support
           0       0.97      0.58      0.73       200
           1       0.43      0.33      0.37       200
           2       0.41      0.04      0.06       200
           3       0.28      1.00      0.44       200
           4       0.00      0.00      0.00       200
    accuracy                           0.39      1000
   macro avg       0.42      0.39      0.32      1000
weighted avg       0.42      0.39      0.32      1000
```
</details>

<details>
<summary>KMeans with TF-IDF</summary>

**Overall Accuracy**: 0.6990
```
              precision    recall  f1-score   support
           0       0.98      0.79      0.87       200
           1       0.59      0.93      0.72       200
           2       0.62      0.84      0.71       200
           3       0.75      0.94      0.83       200
           4       0.00      0.00      0.00       200
    accuracy                           0.70      1000
   macro avg       0.59      0.70      0.63      1000
weighted avg       0.59      0.70      0.63      1000
```
</details>

<details>
<summary>KMeans with Embeddings</summary>

**Overall Accuracy**: 0.7260
```
              precision    recall  f1-score   support
           0       0.77      0.96      0.86       200
           1       0.56      0.90      0.69       200
           2       0.81      0.82      0.82       200
           3       0.83      0.94      0.88       200
           4       0.00      0.00      0.00       200
    accuracy                           0.73      1000
   macro avg       0.59      0.73      0.65      1000
weighted avg       0.59      0.73      0.65      1000
```
</details>

### 2.5. Summary of Single Model Performance

| Algorithm | Bag of Words | TF-IDF | Embeddings |
| :--- | :--- | :--- | :--- |
| **kNN** | 0.3500 | 0.8010 | 0.8590 |
| **MNB** | 0.8710 | 0.8670 | 0.8160 |
| **DT** | 0.6130 | 0.6200 | 0.5110 |
| **KMeans**| 0.3880 | 0.6990 | 0.7260 |

---

## 3. Ensemble Model Benchmarks

### 3.1. Embedding-Only Ensembles (`run_embedding_only_ensembles_on_subset.py`)

*   **Samples per category**: 1000

<details>
<summary>Ensemble 1: MNB(emb) + kNN(emb) (Manual Voting)</summary>

**Accuracy**: 0.8340
```
Classification Report:
              precision    recall  f1-score   support
    astro-ph       0.87      0.98      0.92       200
    cond-mat       0.72      0.95      0.82       200
          cs       0.83      0.94      0.88       200
        math       0.91      0.94      0.92       200
     physics       0.91      0.37      0.53       200
    accuracy                           0.83      1000
   macro avg       0.85      0.83      0.81      1000
weighted avg       0.85      0.83      0.81      1000
```
</details>

<details>
<summary>Ensemble 2: MNB(emb) + kNN(emb) + DT(emb) (Manual Voting)</summary>

**Accuracy**: 0.8280
```
Classification Report:
              precision    recall  f1-score   support
    astro-ph       0.88      0.96      0.92       200
    cond-mat       0.72      0.93      0.81       200
          cs       0.86      0.83      0.85       200
        math       0.89      0.93      0.91       200
     physics       0.81      0.49      0.61       200
    accuracy                           0.83      1000
   macro avg       0.83      0.83      0.82      1000
weighted avg       0.83      0.83      0.82      1000
```
</details>

### 3.2. Heterogeneous Ensembles (`run_heterogenous_ensembles.py`)

*   **Run Date**: 2025-08-21 12:56:29
*   **Samples per category**: 1000

<details>
<summary>Ensemble 1: MNB(bow) + kNN(emb) + DT(tfidf)</summary>

**Overall Accuracy**: 0.8760
```
              precision    recall  f1-score   support
    astro-ph       0.94      0.93      0.93       200
    cond-mat       0.82      0.93      0.87       200
          cs       0.88      0.91      0.89       200
        math       0.94      0.94      0.94       200
     physics       0.80      0.68      0.74       200
    accuracy                           0.88      1000
   macro avg       0.88      0.88      0.87      1000
weighted avg       0.88      0.88      0.87      1000
```
</details>

<details>
<summary>Ensemble 2: MNB(tfidf) + kNN(emb) + DT(bow)</summary>

**Overall Accuracy**: 0.8700
```
              precision    recall  f1-score   support
    astro-ph       0.93      0.96      0.95       200
    cond-mat       0.78      0.95      0.86       200
          cs       0.88      0.92      0.90       200
        math       0.94      0.93      0.93       200
     physics       0.83      0.59      0.69       200
    accuracy                           0.87      1000
   macro avg       0.87      0.87      0.87      1000
weighted avg       0.87      0.87      0.87      1000
```
</details>

<details>
<summary>Ensemble 3: MNB(tfidf) + kNN(emb) + DT(tfidf)</summary>

**Overall Accuracy**: 0.8750
```
              precision    recall  f1-score   support
    astro-ph       0.93      0.95      0.94       200
    cond-mat       0.81      0.94      0.87       200
          cs       0.88      0.91      0.89       200
        math       0.93      0.93      0.93       200
     physics       0.84      0.65      0.73       200
    accuracy                           0.88      1000
   macro avg       0.88      0.88      0.87      1000
weighted avg       0.88      0.88      0.87      1000
```
</details>

<details>
<summary>Ensemble 4: MNB(bow) + kNN(emb) + DT(bow)</summary>

**Overall Accuracy**: 0.8710
```
              precision    recall  f1-score   support
    astro-ph       0.94      0.94      0.94       200
    cond-mat       0.80      0.94      0.86       200
          cs       0.88      0.92      0.89       200
        math       0.94      0.94      0.94       200
     physics       0.80      0.63      0.71       200
    accuracy                           0.87      1000
   macro avg       0.87      0.87      0.87      1000
weighted avg       0.87      0.87      0.87      1000
```
</details>

<details>
<summary>Ensemble 5: MNB(bow) + kNN(emb)</summary>

**Overall Accuracy**: 0.8580
```
              precision    recall  f1-score   support
    astro-ph       0.91      0.96      0.94       200
    cond-mat       0.73      0.96      0.83       200
          cs       0.87      0.94      0.90       200
        math       0.92      0.94      0.93       200
     physics       0.92      0.48      0.64       200
    accuracy                           0.86      1000
   macro avg       0.87      0.86      0.85      1000
weighted avg       0.87      0.86      0.85      1000
```
</details>

<details>
<summary>Ensemble 6: MNB(tfidf) + kNN(emb)</summary>

**Overall Accuracy**: 0.8500
```
              precision    recall  f1-score   support
    astro-ph       0.89      0.97      0.93       200
    cond-mat       0.72      0.97      0.83       200
          cs       0.87      0.94      0.90       200
        math       0.92      0.93      0.93       200
     physics       0.95      0.44      0.60       200
    accuracy                           0.85      1000
   macro avg       0.87      0.85      0.84      1000
weighted avg       0.87      0.85      0.84      1000
```
</details>

### 3.3. Summary of Heterogeneous Ensemble Performance

| Ensemble Configuration | Accuracy |
| :--- | :--- |
| MNB(emb) + kNN(emb) + DT(emb) | 0.8280 |
| MNB(bow) + kNN(emb) + DT(tfidf) | 0.8760 |
| MNB(tfidf) + kNN(emb) + DT(bow) | 0.8700 |
| MNB(tfidf) + kNN(emb) + DT(tfidf)| 0.8750 |
| MNB(bow) + kNN(emb) + DT(bow) | 0.8710 |
| MNB(emb) + kNN(emb) | 0.8340 |
| MNB(bow) + kNN(emb) | 0.8580 |
| MNB(tfidf) + kNN(emb) | 0.8500 |

---

## 4. Stacking Ensemble Benchmarks (`run_stacking_benchmark.py`)

*   **Run Date**: 2025-08-21 13:42:00
*   **Samples per category**: 1000

<details>
<summary>Stack 1: [MNB(b)+kNN(e)+DT(t)] + LR(b)</summary>

**Overall Accuracy**: 0.8870
```
              precision    recall  f1-score   support
    astro-ph       0.97      0.92      0.94       200
    cond-mat       0.85      0.91      0.88       200
          cs       0.91      0.90      0.90       200
        math       0.93      0.98      0.95       200
     physics       0.77      0.73      0.75       200
    accuracy                           0.89      1000
   macro avg       0.89      0.89      0.89      1000
weighted avg       0.89      0.89      0.89      1000
```
</details>

<details>
<summary>Stack 2: [MNB(b)+kNN(e)+DT(t)] + LR(t)</summary>

**Overall Accuracy**: 0.8950
```
              precision    recall  f1-score   support
    astro-ph       0.97      0.94      0.96       200
    cond-mat       0.88      0.92      0.90       200
          cs       0.91      0.89      0.90       200
        math       0.91      0.97      0.94       200
     physics       0.79      0.76      0.78       200
    accuracy                           0.90      1000
   macro avg       0.89      0.89      0.89      1000
weighted avg       0.89      0.90      0.89      1000
```
</details>

<details>
<summary>Stack 3: [MNB(b)+kNN(e)+DT(t)] + LR(e)</summary>

**Overall Accuracy**: 0.8870
```
              precision    recall  f1-score   support
    astro-ph       0.97      0.93      0.95       200
    cond-mat       0.88      0.91      0.89       200
          cs       0.90      0.89      0.89       200
        math       0.91      0.97      0.94       200
     physics       0.77      0.74      0.76       200
    accuracy                           0.89      1000
   macro avg       0.89      0.89      0.89      1000
weighted avg       0.89      0.89      0.89      1000
```
</details>

<details>
<summary>Stack 4: [MNB(t)+kNN(e)+DT(t)] + LR(b)</summary>

**Overall Accuracy**: 0.8820
```
              precision    recall  f1-score   support
    astro-ph       0.97      0.93      0.95       200
    cond-mat       0.83      0.90      0.86       200
          cs       0.93      0.89      0.91       200
        math       0.92      0.98      0.95       200
     physics       0.76      0.72      0.74       200
    accuracy                           0.88      1000
   macro avg       0.88      0.88      0.88      1000
weighted avg       0.88      0.88      0.88      1000
```
</details>

<details>
<summary>Stack 5: [MNB(t)+kNN(e)+DT(t)] + LR(t)</summary>

**Overall Accuracy**: 0.8980
```
              precision    recall  f1-score   support
    astro-ph       0.97      0.93      0.95       200
    cond-mat       0.88      0.92      0.90       200
          cs       0.93      0.89      0.91       200
        math       0.92      0.98      0.95       200
     physics       0.79      0.77      0.78       200
    accuracy                           0.90      1000
   macro avg       0.90      0.90      0.90      1000
weighted avg       0.90      0.90      0.90      1000
```
</details>

<details>
<summary>Stack 6: [MNB(t)+kNN(e)+DT(t)] + LR(e)</summary>

**Overall Accuracy**: 0.8930
```
              precision    recall  f1-score   support
    astro-ph       0.97      0.93      0.95       200
    cond-mat       0.88      0.92      0.90       200
          cs       0.91      0.88      0.89       200
        math       0.92      0.97      0.95       200
     physics       0.78      0.77      0.77       200
    accuracy                           0.89      1000
   macro avg       0.89      0.89      0.89      1000
weighted avg       0.89      0.89      0.89      1000
```
</details>

<details>
<summary>Stack 7: [MNB(b)+kNN(e)+DT(t)] + DT(t)</summary>

**Overall Accuracy**: 0.8570
```
              precision    recall  f1-score   support
    astro-ph       0.95      0.93      0.94       200
    cond-mat       0.83      0.89      0.86       200
          cs       0.88      0.86      0.87       200
        math       0.89      0.92      0.90       200
     physics       0.73      0.69      0.71       200
    accuracy                           0.86      1000
   macro avg       0.86      0.86      0.86      1000
weighted avg       0.86      0.86      0.86      1000
```
</details>

<details>
<summary>Stack 8: [MNB(b)+kNN(e)] + DT(t)</summary>

**Overall Accuracy**: 0.8700
```
              precision    recall  f1-score   support
    astro-ph       0.97      0.93      0.95       200
    cond-mat       0.85      0.86      0.86       200
          cs       0.89      0.86      0.88       200
        math       0.90      0.93      0.92       200
     physics       0.74      0.77      0.75       200
    accuracy                           0.87      1000
   macro avg       0.87      0.87      0.87      1000
weighted avg       0.87      0.87      0.87      1000
```
</details>

<details>
<summary>Stack 9: [MNB(b)+kNN(e)] + LR(b)</summary>

**Overall Accuracy**: 0.8850
```
              precision    recall  f1-score   support
    astro-ph       0.97      0.92      0.94       200
    cond-mat       0.85      0.91      0.87       200
          cs       0.91      0.90      0.90       200
        math       0.93      0.98      0.95       200
     physics       0.77      0.72      0.75       200
    accuracy                           0.89      1000
   macro avg       0.88      0.89      0.88      1000
weighted avg       0.88      0.89      0.88      1000
```
</details>

<details>
<summary>Stack 10: [MNB(b)+kNN(e)] + LR(t)</summary>

**Overall Accuracy**: 0.8910
```
              precision    recall  f1-score   support
    astro-ph       0.97      0.93      0.95       200
    cond-mat       0.88      0.92      0.90       200
          cs       0.91      0.89      0.90       200
        math       0.91      0.97      0.94       200
     physics       0.78      0.76      0.77       200
    accuracy                           0.89      1000
   macro avg       0.89      0.89      0.89      1000
weighted avg       0.89      0.89      0.89      1000
```
</details>

<details>
<summary>Stack 11: [MNB(b)+kNN(e)] + LR(e)</summary>

**Overall Accuracy**: 0.8870
```
              precision    recall  f1-score   support
    astro-ph       0.97      0.93      0.95       200
    cond-mat       0.88      0.91      0.89       200
          cs       0.90      0.89      0.89       200
        math       0.91      0.97      0.94       200
     physics       0.77      0.74      0.76       200
    accuracy                           0.89      1000
   macro avg       0.89      0.89      0.89      1000
weighted avg       0.89      0.89      0.89      1000
```
</details>

### 4.1. Summary of Stacking Ensemble Performance

| Stacking Configuration | Accuracy |
| :--- | :--- |
| [MNB(b)+kNN(e)+DT(t)] + LR(b) | 0.8870 |
| [MNB(b)+kNN(e)+DT(t)] + LR(t) | 0.8950 |
| [MNB(b)+kNN(e)+DT(t)] + LR(e) | 0.8870 |
| [MNB(t)+kNN(e)+DT(t)] + LR(b) | 0.8820 |
| [MNB(t)+kNN(e)+DT(t)] + LR(t) | 0.8980 |
| [MNB(t)+kNN(e)+DT(t)] + LR(e) | 0.8930 |
| [MNB(b)+kNN(e)+DT(t)] + DT(t) | 0.8570 |
| [MNB(b)+kNN(e)] + DT(t) | 0.8700 |
| [MNB(b)+kNN(e)] + LR(b) | 0.8850 |
| [MNB(b)+kNN(e)] + LR(t) | 0.8910 |
| [MNB(b)+kNN(e)] + LR(e) | 0.8870 |

---

## 5. Ultimate Benchmark (`run_ultimate_benchmark.py`)

### 5.1. SciBERT Model (1000 samples/category)

*   **Run Date**: 2025-08-22 11:56:57

<details>
<summary>LogisticRegression on TF-IDF + Embeddings</summary>

**Overall Accuracy**: 0.8690
```
              precision    recall  f1-score   support
    astro-ph       0.94      0.92      0.93       200
    cond-mat       0.86      0.86      0.86       200
          cs       0.92      0.88      0.90       200
        math       0.93      0.95      0.94       200
     physics       0.71      0.73      0.72       200
    accuracy                           0.87      1000
   macro avg       0.87      0.87      0.87      1000
weighted avg       0.87      0.87      0.87      1000
```
</details>

<details>
<summary>Soft Voting Ensemble [MNB(t)+kNN(e)+DT(t)]</summary>

**Overall Accuracy**: 0.8850
```
              precision    recall  f1-score   support
    astro-ph       0.95      0.92      0.93       200
    cond-mat       0.86      0.92      0.89       200
          cs       0.91      0.90      0.90       200
        math       0.92      0.98      0.95       200
     physics       0.77      0.70      0.74       200
    accuracy                           0.89      1000
   macro avg       0.88      0.89      0.88      1000
weighted avg       0.88      0.89      0.88      1000
```
</details>

<details>
<summary>Pure Stacking [MNB(t)+kNN(e)+DT(t)] + LR</summary>

**Overall Accuracy**: 0.8930
```
              precision    recall  f1-score   support
    astro-ph       0.95      0.92      0.94       200
    cond-mat       0.89      0.92      0.91       200
          cs       0.91      0.90      0.90       200
        math       0.94      0.97      0.95       200
     physics       0.77      0.76      0.78       200
    accuracy                           0.90      1000
   macro avg       0.89      0.89      0.89      1000
weighted avg       0.89      0.90      0.89      1000
```
</details>

<details>
<summary>Stacking [Base Models] + GNB(meta+title)</summary>

**Overall Accuracy**: 0.8840
```
              precision    recall  f1-score   support
    astro-ph       0.96      0.92      0.94       200
    cond-mat       0.85      0.93      0.89       200
          cs       0.90      0.90      0.90       200
        math       0.94      0.96      0.95       200
     physics       0.77      0.71      0.74       200
    accuracy                           0.88      1000
   macro avg       0.88      0.88      0.88      1000
weighted avg       0.88      0.88      0.88      1000
```
</details>

<details>
<summary>Confidence-Gated Ensemble [MNB(t) -> kNN(e)]</summary>

**Overall Accuracy**: 0.8590
```
              precision    recall  f1-score   support
    astro-ph       0.94      0.89      0.92       200
    cond-mat       0.85      0.90      0.87       200
          cs       0.86      0.88      0.87       200
        math       0.92      0.95      0.94       200
     physics       0.72      0.67      0.70       200
    accuracy                           0.86      1000
   macro avg       0.86      0.86      0.86      1000
weighted avg       0.86      0.86      0.86      1000
```
</details>

### 5.2. SciBERT Model (2000 samples/category)

*   **Run Date**: 2025-08-22 12:00:31

<details>
<summary>LogisticRegression on TF-IDF + Embeddings</summary>

**Overall Accuracy**: 0.8570
```
              precision    recall  f1-score   support
    astro-ph       0.94      0.93      0.94       400
    cond-mat       0.85      0.82      0.83       400
          cs       0.88      0.84      0.86       400
        math       0.89      0.94      0.91       400
     physics       0.72      0.76      0.74       400
    accuracy                           0.86      2000
   macro avg       0.86      0.86      0.86      2000
weighted avg       0.86      0.86      0.86      2000
```
</details>

<details>
<summary>Soft Voting Ensemble [MNB(t)+kNN(e)+DT(t)]</summary>

**Overall Accuracy**: 0.8850
```
              precision    recall  f1-score   support
    astro-ph       0.98      0.92      0.94       400
    cond-mat       0.89      0.85      0.87       400
          cs       0.91      0.88      0.90       400
        math       0.91      0.96      0.93       400
     physics       0.75      0.82      0.79       400
    accuracy                           0.89      2000
   macro avg       0.89      0.89      0.89      2000
weighted avg       0.89      0.89      0.89      2000
```
</details>

<details>
<summary>Pure Stacking [MNB(t)+kNN(e)+DT(t)] + LR</summary>

**Overall Accuracy**: 0.8850
```
              precision    recall  f1-score   support
    astro-ph       0.97      0.94      0.95       400
    cond-mat       0.90      0.84      0.87       400
          cs       0.91      0.88      0.90       400
        math       0.91      0.95      0.93       400
     physics       0.75      0.84      0.79       400
    accuracy                           0.89      2000
   macro avg       0.89      0.89      0.89      2000
weighted avg       0.89      0.89      0.89      2000
```
</details>

<details>
<summary>Stacking [Base Models] + GNB(meta+title)</summary>

**Overall Accuracy**: 0.8790
```
              precision    recall  f1-score   support
    astro-ph       0.97      0.92      0.94       400
    cond-mat       0.86      0.86      0.86       400
          cs       0.92      0.89      0.90       400
        math       0.93      0.95      0.94       400
     physics       0.74      0.78      0.76       400
    accuracy                           0.88      2000
   macro avg       0.88      0.88      0.88      2000
weighted avg       0.88      0.88      0.88      2000
```
</details>

<details>
<summary>Confidence-Gated Ensemble [MNB(t) -> kNN(e)]</summary>

**Overall Accuracy**: 0.8785
```
              precision    recall  f1-score   support
    astro-ph       0.98      0.90      0.94       400
    cond-mat       0.86      0.85      0.86       400
          cs       0.91      0.88      0.90       400
        math       0.90      0.96      0.93       400
     physics       0.75      0.80      0.77       400
    accuracy                           0.88      2000
   macro avg       0.88      0.88      0.88      2000
weighted avg       0.88      0.88      0.88      2000
```
</details>

---

## 6. Ultimate Benchmark (`run_ultimate_benchmark_e5.py`)

### 6.1. e5-base Model (1000 samples/category)

*   **Run Date**: 2025-08-22 12:13:03

<details>
<summary>LogisticRegression on TF-IDF + Embeddings</summary>

**Overall Accuracy**: 0.8850
```
              precision    recall  f1-score   support
    astro-ph       0.96      0.95      0.96       200
    cond-mat       0.85      0.90      0.88       200
          cs       0.89      0.90      0.90       200
        math       0.92      0.97      0.94       200
     physics       0.79      0.70      0.75       200
    accuracy                           0.89      1000
   macro avg       0.88      0.89      0.88      1000
weighted avg       0.88      0.89      0.88      1000
```
</details>

<details>
<summary>Soft Voting Ensemble [MNB(t)+kNN(e)+DT(t)]</summary>

**Overall Accuracy**: 0.8920
```
              precision    recall  f1-score   support
    astro-ph       0.97      0.94      0.95       200
    cond-mat       0.85      0.93      0.89       200
          cs       0.90      0.92      0.91       200
        math       0.92      0.97      0.95       200
     physics       0.81      0.71      0.76       200
    accuracy                           0.89      1000
   macro avg       0.89      0.89      0.89      1000
weighted avg       0.89      0.89      0.89      1000
```
</details>

<details>
<summary>Pure Stacking [MNB(t)+kNN(e)+DT(t)] + LR</summary>

**Overall Accuracy**: 0.9010
```
              precision    recall  f1-score   support
    astro-ph       0.97      0.94      0.95       200
    cond-mat       0.88      0.91      0.89       200
          cs       0.93      0.90      0.91       200
        math       0.93      0.98      0.95       200
     physics       0.79      0.79      0.79       200
    accuracy                           0.90      1000
   macro avg       0.90      0.90      0.90      1000
weighted avg       0.90      0.90      0.90      1000
```
</details>

<details>
<summary>Stacking [Base Models] + GNB(meta+title)</summary>

**Overall Accuracy**: 0.8900
```
              precision    recall  f1-score   support
    astro-ph       0.97      0.94      0.95       200
    cond-mat       0.85      0.92      0.88       200
          cs       0.91      0.91      0.91       200
        math       0.94      0.96      0.95       200
     physics       0.79      0.73      0.76       200
    accuracy                           0.89      1000
   macro avg       0.89      0.89      0.89      1000
weighted avg       0.89      0.89      0.89      1000
```
</details>

<details>
<summary>Confidence-Gated Ensemble [MNB(t) -> kNN(e)]</summary>

**Overall Accuracy**: 0.8590
```
              precision    recall  f1-score   support
    astro-ph       0.95      0.91      0.93       200
    cond-mat       0.82      0.92      0.86       200
          cs       0.85      0.88      0.86       200
        math       0.92      0.96      0.94       200
     physics       0.75      0.64      0.69       200
    accuracy                           0.86      1000
   macro avg       0.86      0.86      0.86      1000
weighted avg       0.86      0.86      0.86      1000
```
</details>

### 6.2. e5-base Model (2000 samples/category)

*   **Run Date**: 2025-08-22 12:17:30

<details>
<summary>LogisticRegression on TF-IDF + Embeddings</summary>

**Overall Accuracy**: 0.8820
```
              precision    recall  f1-score   support
    astro-ph       0.98      0.93      0.95       400
    cond-mat       0.89      0.85      0.87       400
          cs       0.90      0.88      0.89       400
        math       0.88      0.96      0.92       400
     physics       0.77      0.80      0.78       400
    accuracy                           0.88      2000
   macro avg       0.88      0.88      0.88      2000
weighted avg       0.88      0.88      0.88      2000
```
</details>

<details>
<summary>Soft Voting Ensemble [MNB(t)+kNN(e)+DT(t)]</summary>

**Overall Accuracy**: 0.8870
```
              precision    recall  f1-score   support
    astro-ph       0.97      0.93      0.95       400
    cond-mat       0.91      0.85      0.88       400
          cs       0.90      0.87      0.88       400
        math       0.89      0.97      0.93       400
     physics       0.77      0.82      0.79       400
    accuracy                           0.89      2000
   macro avg       0.89      0.89      0.89      2000
weighted avg       0.89      0.89      0.89      2000
```
</details>

<details>
<summary>Pure Stacking [MNB(t)+kNN(e)+DT(t)] + LR</summary>

**Overall Accuracy**: 0.8895
```
              precision    recall  f1-score   support
    astro-ph       0.98      0.93      0.95       400
    cond-mat       0.92      0.84      0.88       400
          cs       0.90      0.88      0.89       400
        math       0.90      0.96      0.93       400
     physics       0.77      0.83      0.80       400
    accuracy                           0.89      2000
   macro avg       0.89      0.89      0.89      2000
weighted avg       0.89      0.89      0.89      2000
```
</details>

<details>
<summary>Stacking [Base Models] + GNB(meta+title)</summary>

**Overall Accuracy**: 0.8770
```
              precision    recall  f1-score   support
    astro-ph       0.97      0.92      0.94       400
    cond-mat       0.89      0.85      0.87       400
          cs       0.90      0.88      0.89       400
        math       0.90      0.95      0.93       400
     physics       0.74      0.79      0.76       400
    accuracy                           0.88      2000
   macro avg       0.88      0.88      0.88      2000
weighted avg       0.88      0.88      0.88      2000
```
</details>

<details>
<summary>Confidence-Gated Ensemble [MNB(t) -> kNN(e)]</summary>

**Overall Accuracy**: 0.8800
```
              precision    recall  f1-score   support
    astro-ph       0.98      0.92      0.95       400
    cond-mat       0.89      0.85      0.87       400
          cs       0.90      0.87      0.89       400
        math       0.89      0.96      0.92       400
     physics       0.75      0.81      0.78       400
    accuracy                           0.88      2000
   macro avg       0.88      0.88      0.88      2000
weighted avg       0.88      0.88      0.88      2000
```
</details>

---

## 7. Champion Pipeline Benchmarks

### 7.1. SciBERT Model (`run_champion_pipeline.py`)

#### 1000 Samples per Category

*   **Run Date**: 2025-08-22 11:39:16

<details>
<summary>Meta-Learner: LR(TFIDF)</summary>

**Overall Accuracy**: 0.8940
```
              precision    recall  f1-score   support
    astro-ph       0.95      0.93      0.94       200
    cond-mat       0.90      0.92      0.91       200
          cs       0.90      0.90      0.90       200
        math       0.93      0.97      0.95       200
     physics       0.78      0.74      0.76       200
    accuracy                           0.89      1000
   macro avg       0.89      0.89      0.89      1000
weighted avg       0.89      0.89      0.89      1000
```
</details>

<details>
<summary>Meta-Learner: LR(BoW)</summary>

**Overall Accuracy**: 0.8850
```
              precision    recall  f1-score   support
    astro-ph       0.96      0.92      0.94       200
    cond-mat       0.87      0.90      0.88       200
          cs       0.91      0.90      0.90       200
        math       0.93      0.97      0.95       200
     physics       0.76      0.73      0.74       200
    accuracy                           0.89      1000
   macro avg       0.88      0.89      0.88      1000
weighted avg       0.88      0.89      0.88      1000
```
</details>

<details>
<summary>Meta-Learner: LR(Emb)</summary>

**Overall Accuracy**: 0.8780
```
              precision    recall  f1-score   support
    astro-ph       0.95      0.93      0.94       200
    cond-mat       0.86      0.86      0.86       200
          cs       0.91      0.92      0.92       200
        math       0.94      0.94      0.94       200
     physics       0.74      0.74      0.74       200
    accuracy                           0.88      1000
   macro avg       0.88      0.88      0.88      1000
weighted avg       0.88      0.88      0.88      1000
```
</details>

<details>
<summary>Meta-Learner: XGB(TFIDF)</summary>

**Overall Accuracy**: 0.8940
```
              precision    recall  f1-score   support
    astro-ph       0.96      0.93      0.94       200
    cond-mat       0.88      0.91      0.89       200
          cs       0.91      0.93      0.92       200
        math       0.94      0.95      0.95       200
     physics       0.78      0.76      0.77       200
    accuracy                           0.89      1000
   macro avg       0.89      0.89      0.89      1000
weighted avg       0.89      0.89      0.89      1000
```
</details>

<details>
<summary>Meta-Learner: XGB(BoW)</summary>

**Overall Accuracy**: 0.8860
```
              precision    recall  f1-score   support
    astro-ph       0.95      0.93      0.94       200
    cond-mat       0.88      0.91      0.89       200
          cs       0.91      0.91      0.91       200
        math       0.94      0.95      0.94       200
     physics       0.76      0.74      0.75       200
    accuracy                           0.89      1000
   macro avg       0.89      0.89      0.89      1000
weighted avg       0.89      0.89      0.89      1000
```
</details>

<details>
<summary>Meta-Learner: XGB(Emb)</summary>

**Overall Accuracy**: 0.8940
```
              precision    recall  f1-score   support
    astro-ph       0.96      0.92      0.94       200
    cond-mat       0.89      0.93      0.90       200
          cs       0.90      0.91      0.90       200
        math       0.94      0.96      0.95       200
     physics       0.78      0.76      0.77       200
    accuracy                           0.89      1000
   macro avg       0.89      0.89      0.89      1000
weighted avg       0.89      0.89      0.89      1000
```
</details>

<details>
<summary>Meta-Learner: GNB(TFIDF)</summary>

**Overall Accuracy**: 0.7810
```
              precision    recall  f1-score   support
    astro-ph       0.91      0.93      0.92       200
    cond-mat       0.75      0.79      0.77       200
          cs       0.80      0.81      0.80       200
        math       0.89      0.81      0.85       200
     physics       0.57      0.57      0.57       200
    accuracy                           0.78      1000
   macro avg       0.78      0.78      0.78      1000
weighted avg       0.78      0.78      0.78      1000
```
</details>

<details>
<summary>Meta-Learner: GNB(BoW)</summary>

**Overall Accuracy**: 0.7640
```
              precision    recall  f1-score   support
    astro-ph       0.89      0.89      0.89       200
    cond-mat       0.71      0.77      0.74       200
          cs       0.80      0.82      0.81       200
        math       0.88      0.84      0.86       200
     physics       0.53      0.51      0.52       200
    accuracy                           0.76      1000
   macro avg       0.76      0.76      0.76      1000
weighted avg       0.76      0.76      0.76      1000
```
</details>

<details>
<summary>Meta-Learner: GNB(Emb)</summary>

**Overall Accuracy**: 0.8630
```
              precision    recall  f1-score   support
    astro-ph       0.98      0.89      0.93       200
    cond-mat       0.84      0.86      0.85       200
          cs       0.86      0.92      0.89       200
        math       0.95      0.95      0.95       200
     physics       0.70      0.69      0.70       200
    accuracy                           0.86      1000
   macro avg       0.86      0.86      0.86      1000
weighted avg       0.86      0.86      0.86      1000
```
</details>

#### 2000 Samples per Category

<details>
<summary>Meta-Learner: LR(TFIDF)</summary>

**Overall Accuracy**: 0.8920
```
              precision    recall  f1-score   support
    astro-ph       0.97      0.94      0.96       400
    cond-mat       0.91      0.85      0.88       400
          cs       0.91      0.88      0.89       400
        math       0.90      0.97      0.93       400
     physics       0.78      0.83      0.80       400
    accuracy                           0.89      2000
   macro avg       0.89      0.89      0.89      2000
weighted avg       0.89      0.89      0.89      2000
```
</details>

<details>
<summary>Meta-Learner: LR(BoW)</summary>

**Overall Accuracy**: 0.8825
```
              precision    recall  f1-score   support
    astro-ph       0.98      0.93      0.95       400
    cond-mat       0.87      0.83      0.85       400
          cs       0.92      0.87      0.89       400
        math       0.90      0.96      0.93       400
     physics       0.76      0.83      0.79       400
    accuracy                           0.88      2000
   macro avg       0.89      0.88      0.88      2000
weighted avg       0.89      0.88      0.88      2000
```
</details>

<details>
<summary>Meta-Learner: LR(Emb)</summary>

**Overall Accuracy**: 0.8920
```
              precision    recall  f1-score   support
    astro-ph       0.97      0.94      0.95       400
    cond-mat       0.92      0.85      0.88       400
          cs       0.91      0.88      0.89       400
        math       0.90      0.97      0.93       400
     physics       0.78      0.83      0.80       400
    accuracy                           0.89      2000
   macro avg       0.89      0.89      0.89      2000
weighted avg       0.89      0.89      0.89      2000
```
</details>

<details>
<summary>Meta-Learner: XGB(TFIDF)</summary>

**Overall Accuracy**: 0.8940
```
              precision    recall  f1-score   support
    astro-ph       0.97      0.95      0.96       400
    cond-mat       0.92      0.85      0.88       400
          cs       0.90      0.89      0.90       400
        math       0.91      0.94      0.93       400
     physics       0.78      0.84      0.81       400
    accuracy                           0.89      2000
   macro avg       0.90      0.89      0.89      2000
weighted avg       0.90      0.89      0.89      2000
```
</details>

<details>
<summary>Meta-Learner: XGB(BoW)</summary>

**Overall Accuracy**: 0.8915
```
              precision    recall  f1-score   support
    astro-ph       0.97      0.94      0.96       400
    cond-mat       0.91      0.84      0.87       400
          cs       0.91      0.90      0.91       400
        math       0.91      0.95      0.93       400
     physics       0.77      0.82      0.80       400
    accuracy                           0.89      2000
   macro avg       0.89      0.89      0.89      2000
weighted avg       0.89      0.89      0.89      2000
```
</details>

<details>
<summary>Meta-Learner: XGB(Emb)</summary>

**Overall Accuracy**: 0.8975
```
              precision    recall  f1-score   support
    astro-ph       0.97      0.93      0.95       400
    cond-mat       0.93      0.85      0.89       400
          cs       0.90      0.90      0.90       400
        math       0.92      0.95      0.93       400
     physics       0.79      0.84      0.81       400
    accuracy                           0.90      2000
   macro avg       0.90      0.90      0.90      2000
weighted avg       0.90      0.90      0.90      2000
```
</details>

<details>
<summary>Meta-Learner: GNB(TFIDF)</summary>

**Overall Accuracy**: 0.7775
```
              precision    recall  f1-score   support
    astro-ph       0.92      0.92      0.92       400
    cond-mat       0.74      0.79      0.76       400
          cs       0.77      0.77      0.77       400
        math       0.89      0.82      0.86       400
     physics       0.57      0.59      0.58       400
    accuracy                           0.78      2000
   macro avg       0.78      0.78      0.78      2000
weighted avg       0.78      0.78      0.78      2000
```
</details>

<details>
<summary>Meta-Learner: GNB(BoW)</summary>

**Overall Accuracy**: 0.7425
```
              precision    recall  f1-score   support
    astro-ph       0.87      0.91      0.89       400
    cond-mat       0.68      0.79      0.73       400
          cs       0.74      0.76      0.75       400
        math       0.84      0.82      0.83       400
     physics       0.54      0.43      0.48       400
    accuracy                           0.74      2000
   macro avg       0.74      0.74      0.74      2000
weighted avg       0.74      0.74      0.74      2000
```
</details>

<details>
<summary>Meta-Learner: GNB(Emb)</summary>

**Overall Accuracy**: 0.8700
```
              precision    recall  f1-score   support
    astro-ph       0.97      0.94      0.96       400
    cond-mat       0.81      0.90      0.85       400
          cs       0.87      0.90      0.89       400
        math       0.91      0.96      0.94       400
     physics       0.77      0.65      0.70       400
    accuracy                           0.87      2000
   macro avg       0.87      0.87      0.87      2000
weighted avg       0.87      0.87      0.87      2000
```
</details>

---

## 8. Single Model Benchmark (LR & XGBoost) (`run_single_LR_XBG.py`)

### 8.1. 1000 Samples per Category

*   **Run Date**: 2025-08-22 14:12:02

<details>
<summary>LR(BoW)</summary>

**Overall Accuracy**: 0.8580
```
              precision    recall  f1-score   support
    astro-ph       0.95      0.91      0.93       200
    cond-mat       0.83      0.86      0.85       200
          cs       0.89      0.84      0.87       200
        math       0.89      0.98      0.93       200
     physics       0.72      0.69      0.71       200
    accuracy                           0.86      1000
   macro avg       0.86      0.86      0.86      1000
weighted avg       0.86      0.86      0.86      1000
```
</details>

<details>
<summary>LR(TFIDF)</summary>

**Overall Accuracy**: 0.8700
```
              precision    recall  f1-score   support
    astro-ph       0.97      0.91      0.94       200
    cond-mat       0.85      0.90      0.88       200
          cs       0.87      0.88      0.88       200
        math       0.91      0.95      0.93       200
     physics       0.75      0.70      0.73       200
    accuracy                           0.87      1000
   macro avg       0.87      0.87      0.87      1000
weighted avg       0.87      0.87      0.87      1000
```
</details>

<details>
<summary>LR(Emb)</summary>

**Overall Accuracy**: 0.8560
```
              precision    recall  f1-score   support
    astro-ph       0.94      0.96      0.95       200
    cond-mat       0.80      0.88      0.84       200
          cs       0.85      0.89      0.87       200
        math       0.88      0.95      0.92       200
     physics       0.79      0.59      0.68       200
    accuracy                           0.86      1000
   macro avg       0.85      0.86      0.85      1000
weighted avg       0.85      0.86      0.85      1000
```
</details>

<details>
<summary>XGB(BoW)</summary>

**Overall Accuracy**: 0.8230
```
              precision    recall  f1-score   support
    astro-ph       0.94      0.94      0.94       200
    cond-mat       0.82      0.86      0.84       200
          cs       0.81      0.79      0.80       200
        math       0.84      0.94      0.89       200
     physics       0.68      0.59      0.63       200
    accuracy                           0.82      1000
   macro avg       0.82      0.82      0.82      1000
weighted avg       0.82      0.82      0.82      1000
```
</details>

<details>
<summary>XGB(TFIDF)</summary>

**Overall Accuracy**: 0.8270
```
              precision    recall  f1-score   support
    astro-ph       0.94      0.94      0.94       200
    cond-mat       0.80      0.85      0.83       200
          cs       0.83      0.83      0.83       200
        math       0.86      0.93      0.89       200
     physics       0.68      0.58      0.63       200
    accuracy                           0.83      1000
   macro avg       0.82      0.83      0.82      1000
weighted avg       0.82      0.83      0.82      1000
```
</details>

<details>
<summary>XGB(Emb)</summary>

**Overall Accuracy**: 0.8380
```
              precision    recall  f1-score   support
    astro-ph       0.92      0.96      0.94       200
    cond-mat       0.78      0.82      0.80       200
          cs       0.85      0.86      0.86       200
        math       0.90      0.95      0.92       200
     physics       0.71      0.59      0.65       200
    accuracy                           0.84      1000
   macro avg       0.83      0.84      0.83      1000
weighted avg       0.83      0.84      0.83      1000
```
</details>

### 8.2. 2000 Samples per Category

*   **Run Date**: 2025-08-22 14:15:16

<details>
<summary>LR(BoW)</summary>

**Overall Accuracy**: 0.8515
```
              precision    recall  f1-score   support
    astro-ph       0.99      0.90      0.94       400
    cond-mat       0.84      0.81      0.82       400
          cs       0.89      0.83      0.86       400
        math       0.86      0.95      0.91       400
     physics       0.71      0.77      0.74       400
    accuracy                           0.85      2000
   macro avg       0.86      0.85      0.85      2000
weighted avg       0.86      0.85      0.85      2000
```
</details>

<details>
<summary>LR(TFIDF)</summary>

**Overall Accuracy**: 0.8710
```
              precision    recall  f1-score   support
    astro-ph       0.98      0.92      0.95       400
    cond-mat       0.88      0.85      0.86       400
          cs       0.89      0.86      0.87       400
        math       0.86      0.95      0.90       400
     physics       0.76      0.78      0.77       400
    accuracy                           0.87      2000
   macro avg       0.87      0.87      0.87      2000
weighted avg       0.87      0.87      0.87      2000
```
</details>

<details>
<summary>LR(Emb)</summary>

**Overall Accuracy**: 0.8400
```
              precision    recall  f1-score   support
    astro-ph       0.94      0.92      0.93       400
    cond-mat       0.83      0.82      0.82       400
          cs       0.86      0.84      0.85       400
        math       0.85      0.94      0.89       400
     physics       0.72      0.68      0.70       400
    accuracy                           0.84      2000
   macro avg       0.84      0.84      0.84      2000
weighted avg       0.84      0.84      0.84      2000
```
</details>

<details>
<summary>XGB(BoW)</summary>

**Overall Accuracy**: 0.8345
```
              precision    recall  f1-score   support
    astro-ph       0.96      0.90      0.93       400
    cond-mat       0.85      0.79      0.82       400
          cs       0.87      0.83      0.85       400
        math       0.83      0.94      0.88       400
     physics       0.69      0.72      0.70       400
    accuracy                           0.83      2000
   macro avg       0.84      0.83      0.84      2000
weighted avg       0.84      0.83      0.84      2000
```
</details>

<details>
<summary>XGB(TFIDF)</summary>

**Overall Accuracy**: 0.8360
```
              precision    recall  f1-score   support
    astro-ph       0.95      0.88      0.92       400
    cond-mat       0.87      0.81      0.84       400
          cs       0.86      0.81      0.84       400
        math       0.83      0.93      0.88       400
     physics       0.69      0.75      0.72       400
    accuracy                           0.84      2000
   macro avg       0.84      0.84      0.84      2000
weighted avg       0.84      0.84      0.84      2000
```
</details>

<details>
<summary>XGB(Emb)</summary>

**Overall Accuracy**: 0.8350
```
              precision    recall  f1-score   support
    astro-ph       0.94      0.91      0.92       400
    cond-mat       0.81      0.80      0.80       400
          cs       0.86      0.85      0.86       400
        math       0.86      0.93      0.89       400
     physics       0.70      0.69      0.70       400
    accuracy                           0.83      2000
   macro avg       0.83      0.84      0.83      2000
weighted avg       0.83      0.84      0.83      2000
```
</details>

