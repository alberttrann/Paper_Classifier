## Technical Report: A Systematic Approach to Scientific Abstract Classification using Ensemble Methods

*Check `benchmark_results.txt` for detailed benchmark runs throughout the project*

### Executive Summary

This report details the systematic development and evaluation of a machine learning classifier for categorizing scientific paper abstracts from the 2.29 million-sample arXiv dataset. The project began with an exploration of individual models (Decision Tree, k-Nearest Neighbors, Multinomial Naive Bayes, K-Means) across various feature representations (Bag-of-Words, TF-IDF, SBERT Embeddings). Initial benchmarks revealed that no single feature set was universally optimal; traditional models like Multinomial Naive Bayes excelled with sparse, keyword-based features (BoW, accuracy **0.8710**), while modern semantic models like k-Nearest Neighbors performed best with dense embeddings (accuracy **0.8590**).

This insight led to the development of increasingly sophisticated ensemble models. A **heterogeneous hard-voting ensemble**, which paired each base model with its optimal feature set, surpassed any single model, achieving an accuracy of **0.8760**.

The final and most advanced phase involved implementing a **stacking classifier**, where a meta-learner was trained on the out-of-fold predictions of the base models. This approach yielded the project's highest performance. The champion model, a stack of `[MNB(tfidf) + kNN(emb) + DT(tfidf)]` with a `LogisticRegression(tfidf)` meta-learner, achieved a final accuracy of **0.8980**.

The project successfully navigated significant technical hurdles, including memory constraints with large datasets, model compliance issues, and extreme class cardinality, demonstrating that state-of-the-art performance is achieved through a synergy of appropriate feature engineering, strategic model selection, and advanced ensembling techniques.

### 1. Introduction

#### 1.1. Problem Statement
The task was to build a robust multi-class classifier to automatically assign a category to a scientific paper based on the content of its abstract. The project utilized the `UniverseTBD/arxiv-abstracts-large` dataset, containing over 2.29 million samples across more than 30,000 distinct categories.

#### 1.2. Initial Project Strategy
The initial hypothesis was that a "tribrid" ensemble combining a Decision Tree (DT), Multinomial Naive Bayes (MNB), and k-Nearest Neighbors (kNN) would provide a robust solution. The plan was to leverage modern SBERT embeddings as a unified feature representation for all models.

### 2. Methodology & Technical Hurdles

The project's initial scope was challenged by several significant technical hurdles, which necessitated a more pragmatic and targeted methodology.

#### 2.1. Hurdles of Scale and Data Complexity
1.  **Computational Cost:** Generating SBERT embeddings for 2.29M documents was computationally prohibitive for a single run on consumer hardware.
2.  **Memory Constraints:** The full dataset of embeddings (~7GB) and the extreme number of classes (>30,000) led to critical memory errors during model training. The `MultinomialNB` model, for instance, attempted to allocate over 366 GiB of RAM.
3.  **Class Imbalance:** The dataset exhibited a "long-tail" distribution, with thousands of classes containing only a single sample, making stratified splitting impossible.

#### 2.2. Evolved Methodology: Targeted Sampling and Pruning
To address these hurdles, the methodology was refined:
1.  **Targeted Sampling:** Instead of using the full, imbalanced dataset, a balanced subset was created by sampling 1,000-2,000 documents from each of the 5 most common parent categories (`astro-ph`, `cond-mat`, `cs`, `math`, `physics`). This created a high-quality, manageable dataset for rapid and reliable benchmarking.
2.  **Text Preprocessing:** A standard NLP cleaning pipeline was applied to all abstracts, including lowercasing, removal of URLs and punctuation, stop word removal, and lemmatization.

#### 2.3. Feature Engineering
Three distinct feature representations were benchmarked:
1.  **Bag-of-Words (BoW):** This represents each document as a sparse vector of word counts. It's simple and captures keyword presence but ignores word importance and semantics.
2.  **TF-IDF (Term Frequency-Inverse Document Frequency):** An improvement on BoW, TF-IDF weights each word count by its inverse document frequency, giving more importance to words that are distinctive to a document and less to common words.
3.  **SBERT Embeddings:** A modern approach using the `intfloat/multilingual-e5-base` model. This transforms each abstract into a dense, 768-dimensional vector that captures the semantic *meaning* of the text, not just its keywords.

#### 2.4. Core Algorithms: Theoretical Foundations

*   **Decision Tree (DT):** A non-parametric supervised model that creates a tree-like structure of decision rules. It partitions the data based on feature values that best separate the classes, typically by maximizing **Information Gain** or minimizing **Gini Impurity**. While interpretable, its greedy, recursive nature makes it prone to overfitting and unstable in high-dimensional spaces like text.

*   **k-Nearest Neighbors (kNN):** An instance-based "lazy learner." It classifies a new data point by taking a majority vote of its *k* nearest neighbors in the feature space. Its performance is critically dependent on the distance metric. For our SBERT embeddings, we used **Euclidean (L2) distance**, which, for normalized vectors, is directly proportional to **Cosine Similarity**, the preferred metric for semantic vector comparison. To handle the computational cost, we utilized the **FAISS** library for efficient similarity search.

*   **Multinomial Naive Bayes (MNB):** A probabilistic classifier based on **Bayes' Theorem**:
    $P(\text{class} | \text{document}) = \frac{P(\text{document} | \text{class}) \cdot P(\text{class})}{P(\text{document})}$
    Its "naive" assumption is the conditional independence of features (words). It calculates the probability of a document belonging to a class based on the probabilities of the words in that document, learned from the training data. This model is mathematically suited for discrete word counts, explaining its strong performance with BoW and TF-IDF.

*   **K-Means (as a Classifier):** An unsupervised clustering algorithm that aims to partition data into *k* clusters by minimizing the within-cluster sum of squares (inertia). To adapt it for classification, we first clustered the training data. Then, each cluster was assigned the majority class label of the training samples within it. Test samples were then assigned to the nearest cluster, inheriting that cluster's assigned class label.

### 3. Experimental Results & Analysis

#### 3.1. Experiment 1: Individual Model Benchmarking
This experiment tested each of the 4 algorithms against each of the 3 feature sets.
Check `run_single_benchmarks.py` for code script

**Summary Table 1: Individual Model Accuracy**
| Algorithm | Bag of Words | TF-IDF | Embeddings |
| :--- | :--- | :--- | :--- |
| kNN | 0.3500 | 0.8010 | **0.8590** |
| MNB | **0.8710** | 0.8670 | 0.8160 |
| DT | 0.6130 | **0.6200** | 0.5110 |
| KMeans | 0.3880 | 0.6990 | **0.7260** |

**Key Findings:**
*   **No Universal Best Feature Set:** MNB and DT performed best with traditional sparse features (BoW/TF-IDF), while distance-based kNN and K-Means were unusable with BoW but excelled with dense Embeddings.
*   **MNB(bow) as the Standalone Champion:** The classic combination of Multinomial Naive Bayes and Bag-of-Words was the single best-performing model.
*   **kNN(emb) as the Best "Modern" Model:** kNN paired with semantic embeddings was a very strong contender.
*   **DT's Weakness Confirmed:** The Decision Tree was the weakest supervised learner, confirming its unsuitability for this task in isolation.

#### 3.2. Experiment 2: Heterogeneous Voting Ensembles
This experiment tested the hypothesis that an ensemble where each model is paired with its optimal feature set would outperform any single model. A hard-voting (majority rule) approach was used.

Check `run_heterogenous_ensembles.py` for code script. Also check `run_embedding_only_ensembles.py` for the code script of evaluating the 2 initial embedding-only ensembles. [MNB(emb) + kNN(emb) + DTT(emb) & MNB(emb) + kNN(emb)]

**Summary Table 2: Heterogeneous Voting Ensemble Accuracy**
| Ensemble Configuration | Accuracy |
| :--- | :--- |
| MNB(bow) + kNN(emb) + DT(tfidf) | **0.8760** |
| MNB(tfidf) + kNN(emb) + DT(tfidf) | 0.8750 |
| MNB(bow) + kNN(emb) | 0.8580 |

**Key Findings:**
*   **Ensembling Improves Performance:** The top voting ensemble (0.8760) slightly outperformed the best single model (0.8710).
*   **Value of the "Weak" Learner:** Crucially, the ensemble with the Decision Tree performed better than the ensemble without it. This demonstrates a core principle of ensembling: a weak learner can improve overall performance if its errors are uncorrelated with the errors of the stronger models, providing a valuable "dissenting opinion."

#### 3.3. Experiment 3: Advanced Stacking Ensembles
This final experiment implemented a stacking classifier. A meta-learner was trained on the out-of-fold predictions of the base models, combined with original features.

Check `run_stacking_benchmark.py` for code script

**Summary Table 3: Stacking Ensemble Accuracy**
| Stacking Configuration | Accuracy |
| :--- | :--- |
| `[MNB(t)+kNN(e)+DT(t)] + LR(t)` | **0.8980** |
| `[MNB(b)+kNN(e)+DT(t)] + LR(t)` | 0.8950 |
| `[MNB(t)+kNN(e)+DT(t)] + LR(e)` | 0.8930 |
| `[MNB(b)+kNN(e)] + LR(t)` | 0.8910 |

**Key Findings:**
*   **Stacking is the Ultimate Winner:** The best stacking model achieved an accuracy of **0.8980**, significantly outperforming both the best single model and the best voting ensemble. This proves the value of learning how to weigh and combine base model predictions rather than just taking a simple vote.
*   **TF-IDF for the Meta-Learner:** The highest performing configurations were those where the Logistic Regression meta-learner was given TF-IDF features as additional context. This suggests that after seeing the probabilistic outputs of the base models, the meta-learner benefits from having access to the original keyword-based features to make a final, refined decision.
*   **Logistic Regression as an Ideal Meta-Learner:** The simple, linear Logistic Regression outperformed the more complex Decision Tree as a meta-learner, highlighting that for a meta-task, a simpler model that can effectively weigh strong input features is often superior.

#### Recap

```
--- Configuration ---
Categories: ['astro-ph', 'cond-mat', 'cs', 'math', 'physics']
DATASET_NAME = "UniverseTBD/arxiv-abstracts-large" 
TFIDF_MAX_FEATURES = 10000
KNN_N_NEIGHBORS = 5
Device: cuda
-------------------------
--- Single Benchmark Results Summary (Accuracy) ---
Algorithm       | Bag of Words    | TF-IDF          | Embeddings     
---------------------------------------------------------------------
kNN             | 0.3500          | 0.8010          | 0.8590         
MNB             | 0.8710          | 0.8670          | 0.8160         
DT              | 0.6130          | 0.6200          | 0.5110         
KMeans          | 0.3880          | 0.6990          | 0.7260         
---------------------------------------------------------------------

--- Heterogeneous Ensemble Summary (Accuracy) ---
Ensemble Configuration                             | Accuracy       
--------------------------------------------------------------------
MNB(emb) + kNN(emb) + DT(emb)                      | 0.8280
MNB(bow) + kNN(emb) + DT(tfidf)                    | 0.8760         
MNB(tfidf) + kNN(emb) + DT(bow)                    | 0.8700         
MNB(tfidf) + kNN(emb) + DT(tfidf)                  | 0.8750         
MNB(bow) + kNN(emb) + DT(bow)                      | 0.8710
MNB(emb) + kNN(emb)                                | 0.8340
MNB(bow) + kNN(emb)                                | 0.8580         
MNB(tfidf) + kNN(emb)                              | 0.8500         
--------------------------------------------------------------------

--- Stacking Ensemble Summary (Accuracy) ---
Stacking Configuration                            | Accuracy
--------------------------------------------------|
[MNB(b)+kNN(e)+DT(t)] + LR(b)                     | 0.8870
[MNB(b)+kNN(e)+DT(t)] + LR(t)                     | 0.8950
[MNB(b)+kNN(e)+DT(t)] + LR(e)                     | 0.8870
[MNB(t)+kNN(e)+DT(t)] + LR(b)                     | 0.8820
[MNB(t)+kNN(e)+DT(t)] + LR(t)                     | 0.8980
[MNB(t)+kNN(e)+DT(t)] + LR(e)                     | 0.8930
[MNB(b)+kNN(e)+DT(t)] + DT(t)                     | 0.8570
[MNB(b)+kNN(e)] + DT(t)                           | 0.8700
[MNB(b)+kNN(e)] + LR(b)                           | 0.8850
[MNB(b)+kNN(e)] + LR(t)                           | 0.8910
[MNB(b)+kNN(e)] + LR(e)			                  | 0.8870
```


### **Project Phase 3: Advanced Optimization and Hybrid Modeling**

**Core Principle:** All experiments will be conducted on the targeted, balanced subset of data to ensure rapid iteration and clear, comparable results. We will build a new, comprehensive benchmarking script that executes this entire plan.

---

### **Phase 3, Step 1: Foundational Enhancements (Upgrading the "Ingredients")**

**(No changes here - this step is solid and foundational for everything else)**

*   **Sub-step 1.1:** Enhance Text Cleaning (Custom, Domain-Specific Stop Words).
*   **Sub-step 1.2:** Enhance TF-IDF Vectorizer (n-grams, `min_df`, `max_df`, `sublinear_tf`).
*   **Sub-step 1.3:** Enhance SBERT Embeddings (Switch to **SciBERT**).
*   **Sub-step 1.4:** Hyperparameter Tuning of Base Models (`MNB`, `DT`, `kNN`) using `GridSearchCV`.

---

### **Phase 3, Step 2: Advanced Feature Engineering (Creating New Signals)**

We will now create a diverse portfolio of feature sets.

*   **Sub-step 2.1:** Engineer Structural & Metadata Features (`X_meta`).
*   **Sub-step 2.2:** Engineer "Abstract vs. Title" Features (`X_title_similarity`, `X_title_diff`).
*   **Sub-step 2.3:** Engineer the "Semantic Dissonance" Feature (`X_dissonance`).
*   **Sub-step 2.4: Implement Combined Feature Sets at the Input Layer.**
    *   **Action:** Create new, combined feature matrices before any models are trained.
    *   **Plan:**
        1.  Create `X_tfidf_plus_emb`: Horizontally stack the "advanced" TF-IDF matrix and the SciBERT embeddings matrix using `scipy.sparse.hstack`.
        2.  Create `X_tfidf_plus_meta`: Horizontally stack the "advanced" TF-IDF matrix and the metadata features (`X_meta`) from Step 2.1.
    *   **Goal:** Create two powerful, wide feature sets to test if a single strong model (`XGBoost` or `LogisticRegression`) can outperform ensembles when given access to all features at once.

---

### **Phase 3, Step 3: Advanced Single Model Benchmarks**

Before moving to the final ensembles, we'll test our new combined feature sets.

*   **Sub-step 3.1: Benchmark Models on Combined Features.**
    *   **Action:** Train and evaluate powerful single models on the new feature sets from Step 2.4.
    *   **Plan:**
        1.  Train and test a `LogisticRegression` on `X_tfidf_plus_emb`.
        2.  Train and test an `XGBClassifier` on `X_tfidf_plus_emb`.
        3.  Train and test an `XGBClassifier` on `X_tfidf_plus_meta`.
*   **Goal:** To establish a new "state-of-the-art" single model baseline. It's possible one of these combinations might be so powerful it rivals the ensembles.

---

### **Phase 3, Step 4: Advanced Ensemble and Stacking Benchmarks**

This is the final, comprehensive bake-off.

*   **Sub-step 4.1: Implement Probability Calibration.**
    *   **Action:** Create calibrated versions of our best-tuned base models.
    *   **Plan:**
        1.  Wrap the tuned `DecisionTreeClassifier` in `sklearn.calibration.CalibratedClassifierCV`.
        2.  Wrap the tuned `KNeighborsClassifier` in `CalibratedClassifierCV`. (MNB is generally well-calibrated, so we can skip it initially).
    *   **Goal:** To have "calibrated" versions of our base models ready for use in the soft voting and stacking experiments below.

*   **Sub-step 4.2: Implement Soft Voting with Calibrated Models.**
    *   **Action:** Create a heterogeneous ensemble using soft voting.
    *   **Plan:**
        1.  Use the **calibrated** base models (`CalibratedDT`, `CalibratedKNN`) and the tuned `MNB`.
        2.  Generate `predict_proba` outputs from each and average them to get the final prediction.
*   **Goal:** Test if probability calibration improves the performance of the soft voting ensemble compared to hard voting.

*   **Sub-step 4.3: Implement "Pure" Stacking (Calibrated Probabilities Only).**
    *   **Action:** Test the stacking architecture where the meta-learner is trained *only* on the out-of-fold predictions of the **calibrated** base models.
    *   **Plan:**
        *   Generate out-of-fold `predict_proba` outputs using the calibrated base models.
        *   Train a meta-learner (e.g., `LogisticRegression`) on *only* these probability vectors.
*   **Goal:** Test the "purest" form of stacking with the highest quality probability signals.
        * Instead of:

    ```python
    # The predictions PLUS the original TF-IDF features
    meta_learner_train_X = hstack([
        meta_features_train['MNB_tfidf'], 
        meta_features_train['kNN_emb'], 
        meta_features_train['DT_tfidf'],
        X_train_tfidf  # <-- The "Raw Evidence"
    ]).tocsr()
    ```

    We'll just do:

    ```python 
    # Only the predictions from the base models
    meta_features_pure_train = np.hstack([
        meta_features_train['MNB_tfidf'], 
        meta_features_train['kNN_emb'], 
        meta_features_train['DT_tfidf']
    ])
    ```

*   **Sub-step 4.4: Implement Stacking with New Meta-Learners and All Engineered Features.**
    *   **Action:** Run a new, expanded set of stacking experiments using the **calibrated** base models.
    *   **Plan:**
        1.  **New Meta-Learners:** Benchmark `GaussianNB`
        2.  **New Meta-Features:** For each meta-learner, test its performance when given different combinations of our engineered features from Step 2 (e.g., calibrated probabilities + metadata, calibrated probabilities + title similarity, calibrated probabilities + dissonance).
*   **Goal:** Systematically find the absolute best combination of calibrated base model predictions, original features, and meta-learner.

---

### **Phase 3, Step 5: The "YOLO" Experiments**

These are high-risk, high-reward experiments to be run in parallel or after the main benchmarks.

*   **Sub-step 5.1: Implement the "Confidence-Gated Ensemble."**
    *   **Action:** Build the cascading ensemble where a meta-learner acts as a "reliability gate."
    *   **Plan:**
        1.  Train the tuned `MNB(tfidf)` as the primary model.
        2.  Train a `LogisticRegression` "gatekeeper" to predict if MNB will be correct.
        3.  Implement the inference logic to escalate to the tuned, **calibrated** `kNN(SciBERT)` when the gatekeeper is not confident.
*   **Goal:** Test if this dynamic, efficiency-focused ensemble can match or beat the accuracy of the more complex stacking models.

*   **Sub-step 5.2: Implement the "Model Chimera" - kNN-Informed Decision Tree.**
    *   **Action:** Attempt a proof-of-concept implementation of this hybrid model. This is a research-level task.
    *   **Plan:**
        1.  Define a custom Python class for the `KnnInformedDecisionTree`.
        2.  The `fit` method will recursively build the tree.
        3.  The `_find_best_split` method at each node will **not** iterate through features. Instead, it will:
            *   Run a localized kNN search on the data points currently at that node.
            *   Generate meta-features like `neighbor_purity` for each point.
            *   Find the best threshold to split the node based on one of these meta-features (e.g., `if neighbor_purity <= 0.6`).
        4.  The `predict` method will traverse this custom tree.
*   **Goal:** Explore a truly novel modeling architecture. The primary outcome is the learning experience and insight, with a potential (but not guaranteed) for high performance.

---

