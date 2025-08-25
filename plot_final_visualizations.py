# create_final_visualizations.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Final, Curated Results Data ---
# These are the top-performing models from each major phase of the project,
# evaluated on the 8-category, single-label test set.

data = {
    "Model Architecture": [
        "Single Model: MNB (TF-IDF)",
        "Single Model: kNN (e5-Emb)",
        "Single Model: LR (TF-IDF)",
        "Voting Ensemble (Hard)",
        "Stacking Ensemble",
        "Fine-Tuned Transformer (e5)"
    ],
    "Abbreviation": [
        "MNB",
        "kNN",
        "LR",
        "Voting",
        "Stacking",
        "Fine-Tuned e5"
    ],
    "Accuracy": [
        0.8596,  # From Grand Champion: Tuned MNB(tfidf)
        0.8599,  # From Grand Champion: Tuned kNN(emb)
        0.8684,  # From Grand Champion: LR(tfidf)
        0.8760,  # From Heterogeneous Ensemble 1 (Best Voting)
        0.8956,  # From Grand Champion: Stack: LR(TFIDF)
        0.9135   # From evaluating the fine-tuned model on the single-label set
    ],
    "Type": [
        "Single Model",
        "Single Model",
        "Single Model",
        "Ensemble",
        "Ensemble",
        "Deep Learning"
    ]
}

df = pd.DataFrame(data)

# Sort by accuracy for plotting
df_sorted = df.sort_values("Accuracy", ascending=False)

# --- Plotting ---

# Set plot style and color palette
sns.set_theme(style="whitegrid")
palette = sns.color_palette("viridis", n_colors=len(df))

# 1. Main Bar Plot: The Final Hierarchy
plt.figure(figsize=(12, 8))
ax1 = sns.barplot(x="Accuracy", y="Model Architecture", data=df_sorted, palette=palette, orient='h')

ax1.set_title("Final Model Performance on Single-Label Classification (8 Categories)", fontsize=18, weight='bold', pad=20)
ax1.set_xlabel("Overall Accuracy", fontsize=14)
ax1.set_ylabel("Model Type", fontsize=14)
ax1.set_xlim(0.85, 0.925) # Zoom in on the top performers

# Add value labels to the bars
for p in ax1.patches:
    width = p.get_width()
    ax1.text(width + 0.001, p.get_y() + p.get_height() / 2,
             f'{width:.4f}',
             va='center', fontsize=12)

plt.tight_layout()
plt.savefig("final_model_hierarchy.png", dpi=300)
print("Saved final_model_hierarchy.png")


# 2. Stepped Bar Plot: The Journey of Improvement
plt.figure(figsize=(10, 7))
df_journey = df.sort_values("Accuracy", ascending=True)
ax2 = sns.barplot(x="Abbreviation", y="Accuracy", data=df_journey, hue="Type", palette="magma", dodge=False)

ax2.set_title("Project Journey: Performance Gains at Each Stage", fontsize=18, weight='bold', pad=20)
ax2.set_xlabel("Model Architecture", fontsize=14)
ax2.set_ylabel("Overall Accuracy", fontsize=14)
ax2.set_ylim(0.85, 0.925) # Zoom in

# Add value labels
for p in ax2.patches:
    ax2.annotate(f'{p.get_height():.4f}',
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center',
                   xytext=(0, 9),
                   textcoords='offset points', fontsize=11, weight='bold')

plt.tight_layout()
plt.savefig("project_journey_performance.png", dpi=300)
print("Saved project_journey_performance.png")


# 3. Detailed F1-Score Comparison: Stacking vs. Fine-Tuned
# Manually enter the F1-scores for the two champions
categories = ['math', 'astro-ph', 'cs', 'cond-mat', 'physics', 'hep-ph', 'quant-ph', 'hep-th']
f1_scores = {
    'Stacking Ensemble': [0.93, 0.96, 0.91, 0.86, 0.76, 0.94, 0.88, 0.92],
    'Fine-Tuned e5':    [0.91, 0.98, 0.91, 0.91, 0.81, 0.95, 0.91, 0.93]
}

df_f1 = pd.DataFrame(f1_scores, index=categories)
df_f1_melted = df_f1.reset_index().melt('index', var_name='Model', value_name='F1-Score')

plt.figure(figsize=(14, 8))
ax3 = sns.barplot(x="index", y="F1-Score", hue="Model", data=df_f1_melted, palette="rocket")

ax3.set_title("Per-Category F1-Score: Stacking vs. Fine-Tuned Model", fontsize=18, weight='bold', pad=20)
ax3.set_xlabel("Scientific Category", fontsize=14)
ax3.set_ylabel("F1-Score", fontsize=14)
ax3.set_ylim(0.7, 1.0)
plt.xticks(rotation=45, ha='right')

# Add value labels
for p in ax3.patches:
    ax3.annotate(f'{p.get_height():.2f}',
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center',
                   xytext=(0, 9),
                   textcoords='offset points', fontsize=10)

plt.legend(title='Model Architecture', fontsize=12)
plt.tight_layout()
plt.savefig("champion_f1_score_comparison.png", dpi=300)
print("Saved champion_f1_score_comparison.png")

# Display plots
plt.show()