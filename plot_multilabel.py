
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from adjustText import adjust_text

# --- Data from All Benchmark Runs ---

# Data from run_multilabel_championship.py
data_championship = {
    "Model": [
        "Tier 1: BR with LR(tfidf)",
        "Tier 2: BR with Soft Voting",
        "Tier 3: BR with Stacking"
    ],
    "Subset Accuracy": [0.5624, 0.6895, 0.6362],
    "Hamming Loss": [0.0747, 0.0518, 0.0606]
}

# Data from run_multilabel_adaptation_benchmark.py
data_adaptation = {
    "Model": [
        "ClassifierChain(RF(TF-IDF))",
        "ClassifierChain(RF(Emb))"
    ],
    "Subset Accuracy": [0.6150, 0.3419],
    "Hamming Loss": [0.0708, 0.1042]
}

# Data from run_multilabel_grand_prix.py
data_grand_prix = {
    "Model": [
        "kNN(Emb)", "VoteEns_2", "VoteEns_1", "XGB(TFIDF)", "XGB(BoW)",
        "Stack_LR(t)", "Pure_Stack_LR", "XGB(Emb)", "LR(TFIDF)", "LR(BoW)",
        "LR(Emb)", "GNB(Emb)", "GNB(TFIDF)", "GNB(BoW)", "KMeans(Emb)"
    ],
    "Subset Accuracy": [
        0.7825, 0.7712, 0.7603, 0.7138, 0.7104, 0.7104, 0.6902, 0.6895,
        0.6490, 0.6451, 0.4835, 0.3819, 0.3151, 0.2701, 0.1670
    ],
    "Hamming Loss": [
        0.0394, 0.0387, 0.0405, 0.0476, 0.0479, 0.0475, 0.0500, 0.0536,
        0.0582, 0.0630, 0.0928, 0.1145, 0.1920, 0.2506, 0.1321
    ]
}

# --- Create Pandas DataFrames ---
df_champ = pd.DataFrame(data_championship)
df_adapt = pd.DataFrame(data_adaptation)
df_gp = pd.DataFrame(data_grand_prix)

# Combine all results into a single DataFrame
df_all = pd.concat([df_champ, df_adapt, df_gp], ignore_index=True)

# --- Plotting ---

# Set plot style
sns.set_theme(style="whitegrid")

# 1. Bar Plot for Subset Accuracy (The Main Result)
plt.figure(figsize=(12, 10))
df_sorted_acc = df_all.sort_values("Subset Accuracy", ascending=False)
ax1 = sns.barplot(x="Subset Accuracy", y="Model", data=df_sorted_acc, palette="viridis")
ax1.set_title("Multi-Label Classification: Subset Accuracy Comparison", fontsize=16, weight='bold')
ax1.set_xlabel("Subset Accuracy (Higher is Better)", fontsize=12)
ax1.set_ylabel("Model Architecture", fontsize=12)

# Add value labels to the bars
for p in ax1.patches:
    width = p.get_width()
    ax1.text(width + 0.01, p.get_y() + p.get_height() / 2,
             f'{width:.4f}',
             va='center')

plt.xlim(0, 1.0)
plt.tight_layout()
plt.savefig("subset_accuracy_comparison.png", dpi=300)
print("Saved subset_accuracy_comparison.png")


# 2. Bar Plot for Hamming Loss (The Reliability Result)
plt.figure(figsize=(12, 10))
df_sorted_loss = df_all.sort_values("Hamming Loss", ascending=True)
ax2 = sns.barplot(x="Hamming Loss", y="Model", data=df_sorted_loss, palette="plasma")
ax2.set_title("Multi-Label Classification: Hamming Loss Comparison", fontsize=16, weight='bold')
ax2.set_xlabel("Hamming Loss (Lower is Better)", fontsize=12)
ax2.set_ylabel("Model Architecture", fontsize=12)

# Add value labels to the bars
for p in ax2.patches:
    width = p.get_width()
    ax2.text(width + 0.005, p.get_y() + p.get_height() / 2,
             f'{width:.4f}',
             va='center')

plt.tight_layout()
plt.savefig("hamming_loss_comparison.png", dpi=300)
print("Saved hamming_loss_comparison.png")

# 3. Scatter Plot to show trade-off between the two metrics
plt.figure(figsize=(14, 10))
ax3 = sns.scatterplot(x="Hamming Loss", y="Subset Accuracy", data=df_all, hue="Model", s=30, legend=False, palette="tab20")
ax3.set_title("Model Performance: Subset Accuracy vs. Hamming Loss", fontsize=16, weight='bold')
ax3.set_xlabel("Hamming Loss (Lower is Better)", fontsize=12)
ax3.set_ylabel("Subset Accuracy (Higher is Better)", fontsize=12)

# Create text annotations
texts = []
for i in range(df_all.shape[0]):
    texts.append(plt.text(df_all["Hamming Loss"][i], df_all["Subset Accuracy"][i], df_all["Model"][i], fontsize=9))

# Automatically adjust text to avoid overlap
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))

# Highlight the "sweet spot"
plt.axvline(x=0.06, color='grey', linestyle='--')
plt.axhline(y=0.7, color='grey', linestyle='--')

plt.tight_layout()
plt.savefig("accuracy_vs_loss_tradeoff.png", dpi=300)
print("Saved accuracy_vs_loss_tradeoff.png")

plt.show()