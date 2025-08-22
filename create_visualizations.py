# create_final_visualizations.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec

# --- Style and Palette Configuration ---
sns.set_theme(style="whitegrid")
# Use the 'constrained' layout engine, which is more robust and handles colorbars correctly.
plt.rcParams['figure.constrained_layout.use'] = True
palette_features = "viridis"
palette_ensembles = "plasma"
palette_embedding = "coolwarm"
palette_scale = "magma"
palette_category = "magma"
palette_meta = "crest"

# --- Data Preparation ---
# === Chart 1 & 2 Data: Single Model Performance ===
df_single_all = pd.DataFrame({
    'Algorithm': ['kNN', 'kNN', 'kNN', 'MNB', 'MNB', 'MNB', 'DT', 'DT', 'DT', 'KMeans', 'KMeans', 'KMeans',
                  'LR', 'LR', 'LR', 'XGB', 'XGB', 'XGB'],
    'Feature Type': ['BoW', 'TF-IDF', 'Embeddings']*6,
    'Accuracy': [0.3500, 0.8010, 0.8590, 0.8710, 0.8670, 0.8160, 0.6130, 0.6200, 0.5110, 0.3880, 0.6990, 0.7260,
                 0.8515, 0.8710, 0.8400, 0.8345, 0.8360, 0.8350]
})
# === Chart 3 Data: Evolution of Ensemble Performance ===
df_ensemble_perf = pd.DataFrame({
    'Ensemble Stage': [
        'A. Best Single Model', 'B. Homogeneous Ensemble', 'C. Heterogeneous Voting', 
        'D. Heterogeneous Stacking', 'E. Champion Pipeline'
    ],
    'Description': [
        'MNB(BoW)', 'MNB+kNN+DT (All Emb)', 'MNB(b)+kNN(e)+DT(t)', 
        '[MNB(t)+kNN(e)+DT(t)] + LR(t)', 'Champion Stack with e5-base'
    ],
    'Accuracy': [0.8710, 0.8280, 0.8760, 0.8980, 0.9040]
})
df_ensemble_perf['Full Description'] = df_ensemble_perf['Ensemble Stage'].str.split('. ').str[1] + "\n(" + df_ensemble_perf['Description'] + ")"

# === Chart 4 & 5 Data: Stacking Meta-Learner Analysis (e5-base, 1k samples/cat) ===
df_stacking_meta_e5 = pd.DataFrame({
    'Meta-Learner': ['LR', 'LR', 'LR', 'XGB', 'XGB', 'XGB', 'GNB', 'GNB', 'GNB'],
    'Meta Features': ['TFIDF', 'BoW', 'Emb', 'TFIDF', 'BoW', 'Emb', 'TFIDF', 'BoW', 'Emb'],
    'Accuracy': [0.9040, 0.8840, 0.9020, 0.9020, 0.9030, 0.9030, 0.7810, 0.7640, 0.8700]
})
# === Chart 6 & 7 Data: SciBERT vs e5-base Head-to-Head ===
df_embedding_comparison = pd.DataFrame({
    'Model Type': ['kNN (Single)', 'Pure Stack', 'Champ Stack LR(t)', 'Champ Stack XGB(e)'],
    'SciBERT': [0.8590, 0.8850, 0.8900, 0.8880],
    'e5-base': [0.8590, 0.8895, 0.8920, 0.8975]
})
df_embedding_melt = pd.melt(df_embedding_comparison, id_vars=['Model Type'], var_name='Embedding', value_name='Accuracy')
# === Chart 8 Data: Impact of Data Scale ===
df_scale_impact = pd.DataFrame({
    'Model': ['MNB(TFIDF)', 'kNN(e5-Emb)', 'Hetero. Voting', 'Champ. Stack LR(t, e5)', 'Champ. Stack XGB(e, e5)'],
    '1000 samples/cat': [0.8670, 0.8590, 0.8760, 0.9040, 0.9030],
    '2000 samples/cat': [0.8710, 0.8400, 0.8760, 0.8920, 0.8975]
})
df_scale_melt = pd.melt(df_scale_impact, id_vars=['Model'], var_name='Sample Size', value_name='Accuracy')
# === Data for Champion Model Breakdown ===
df_champion_report = pd.DataFrame({
    'Category': ['astro-ph', 'cond-mat', 'cs', 'math', 'physics'],
    'Precision': [0.97, 0.89, 0.93, 0.92, 0.81],
    'Recall': [0.94, 0.91, 0.90, 0.98, 0.79],
    'F1-Score': [0.96, 0.90, 0.91, 0.95, 0.80]
})
# === Data for Chart 11 ===
df_f1_comparison = pd.DataFrame({
    'Model': [
        'Single MNB(BoW)', 'Single LR(TFIDF)', 'Voting Hetero.', 
        'Stacking [e5, LR(t)]', 'Stacking [e5, XGB(e)]'
    ],
    'astro-ph': [0.93, 0.94, 0.93, 0.96, 0.96],
    'cond-mat': [0.88, 0.88, 0.87, 0.90, 0.90],
    'cs': [0.89, 0.88, 0.89, 0.91, 0.91],
    'math': [0.94, 0.93, 0.94, 0.95, 0.95],
    'physics': [0.71, 0.73, 0.74, 0.80, 0.79]
})
df_f1_melt = pd.melt(df_f1_comparison, id_vars='Model', var_name='Category', value_name='F1-Score')

# --- Plotting ---
# --- MODIFIED: Use constrained_layout=True in figure creation, remove rcParams ---
fig = plt.figure(figsize=(24, 32), constrained_layout=True)
gs = GridSpec(4, 3, figure=fig)
fig.suptitle('Comprehensive Analysis of Scientific Abstract Classification Models', fontsize=32, fontweight='bold')

# --- Row 1: Foundational Benchmarks ---
ax1 = fig.add_subplot(gs[0, 0])
sns.barplot(x='Algorithm', y='Accuracy', hue='Feature Type', data=df_single_all, ax=ax1, palette=palette_features)
ax1.set_title('Chart 1: Single Model Performance', fontsize=18, fontweight='bold')
ax1.set_ylim(0, 1.0); ax1.set_ylabel('Accuracy', fontsize=14); ax1.set_xlabel('Algorithm', fontsize=14)
ax1.legend(title='Feature Type', fontsize=10); ax1.tick_params(axis='x', rotation=30)

ax2 = fig.add_subplot(gs[0, 1])
df_feature_pivot = df_single_all.pivot_table(index='Algorithm', columns='Feature Type', values='Accuracy')[['BoW', 'TF-IDF', 'Embeddings']]
sns.heatmap(df_feature_pivot, annot=True, cmap='viridis', fmt='.4f', linewidths=.5, ax=ax2)
ax2.set_title('Chart 2: Feature Effectiveness by Model', fontsize=18, fontweight='bold')
ax2.set_ylabel('Algorithm', fontsize=14); ax2.set_xlabel('Feature Type', fontsize=14)

ax3 = fig.add_subplot(gs[0, 2])
# Use the new 'Full Description' for the y-axis
bars = sns.barplot(x='Accuracy', y='Full Description', hue='Ensemble Stage', data=df_ensemble_perf, ax=ax3, palette=palette_ensembles, dodge=False, legend=False)
ax3.set_title('Chart 3: The Path to 90.4% Accuracy', fontsize=18, fontweight='bold')
ax3.set_xlim(0.82, 0.92)
ax3.set_xlabel('Accuracy', fontsize=14)
ax3.set_ylabel('Model Configuration', fontsize=14)

# New, robust annotation logic
for bar in bars.patches:
    y = bar.get_y() + bar.get_height() / 2
    x = bar.get_width()
    
    # Define a threshold for placing text inside or outside
    threshold = 0.84 
    
    if x > threshold:
        # Place text inside the bar, aligned right
        ax3.text(x - 0.002, y, f'{x:.4f}', ha='right', va='center', color='white', fontweight='bold', fontsize=12)
    else:
        # Place text outside the bar, aligned left
        ax3.text(x + 0.001, y, f'{x:.4f}', ha='left', va='center', color='black', fontweight='bold', fontsize=12)

# --- Row 2: Deep Dive into Stacking ---
ax4 = fig.add_subplot(gs[1, 0])
sns.barplot(x='Accuracy', y='Meta-Learner', hue='Meta Features', data=df_stacking_meta_e5, ax=ax4, dodge=True, palette="crest")
ax4.set_title('Chart 4: Impact of Meta-Learner Features\n(Stacking with e5-base @ 1k)', fontsize=18, fontweight='bold')
ax4.set_xlim(0.7, 1.0); ax4.set_xlabel('Accuracy', fontsize=14); ax4.set_ylabel('Meta-Learner Algorithm', fontsize=14)
ax4.legend(title='Meta Features', loc='lower right')

ax5 = fig.add_subplot(gs[1, 1])
df_meta_best = df_stacking_meta_e5.loc[df_stacking_meta_e5.groupby('Meta-Learner')['Accuracy'].idxmax()]
sns.barplot(x='Accuracy', y='Meta-Learner', data=df_meta_best.sort_values('Accuracy', ascending=False), ax=ax5, hue='Meta-Learner', palette='crest', legend=False)
ax5.set_title('Chart 5: Peak Performance by Meta-Learner', fontsize=18, fontweight='bold')
ax5.set_xlim(0.7, 1.0); ax5.set_xlabel('Best Achieved Accuracy', fontsize=14); ax5.set_ylabel('Meta-Learner Algorithm', fontsize=14)
for p in ax5.patches:
    ax5.text(p.get_width(), p.get_y() + p.get_height() / 2, f' {p.get_width():.4f}', va='center')

ax6 = fig.add_subplot(gs[1, 2])
sns.barplot(x='Model Type', y='Accuracy', hue='Embedding', data=df_embedding_melt, ax=ax6, palette=palette_embedding)
ax6.set_title('Chart 6: SciBERT vs. e5-base Head-to-Head', fontsize=18, fontweight='bold')
ax6.set_ylim(0.85, 0.91); ax6.set_ylabel('Accuracy', fontsize=14); ax6.set_xlabel('Model Architecture', fontsize=14)
plt.setp(ax6.get_xticklabels(), rotation=30, ha='right')

# --- Row 3: Deep Dive into Other Factors ---
ax7 = fig.add_subplot(gs[2, 0])
sns.barplot(x='Model', y='Accuracy', hue='Sample Size', data=df_scale_melt, ax=ax7, palette=palette_scale)
ax7.set_title('Chart 7: Impact of Data Scale (1k vs 2k)', fontsize=18, fontweight='bold')
ax7.set_ylim(0.83, 0.92)
ax7.set_ylabel('Accuracy', fontsize=14)
ax7.set_xlabel('')
plt.setp(ax7.get_xticklabels(), rotation=30, ha='right')
ax7.legend(title='Samples per Category')

ax8 = fig.add_subplot(gs[2, 1:])
top_single_models = df_single_all.sort_values('Accuracy', ascending=False).head(5)
top_ensembles = df_ensemble_perf.iloc[2:].sort_values('Accuracy', ascending=False)
top_ensembles.rename(columns={'Ensemble Stage': 'Model Name', 'Description': 'Configuration'}, inplace=True)
top_single_models.rename(columns={'Algorithm': 'Model Name', 'Feature Type': 'Configuration'}, inplace=True)
df_finalists = pd.concat([top_single_models[['Configuration', 'Accuracy']], top_ensembles[['Configuration', 'Accuracy']]])
sns.barplot(x='Accuracy', y='Configuration', data=df_finalists, ax=ax8, hue='Configuration', palette='cividis', legend=False, errorbar=None)
ax8.set_title('Chart 8: The Finalists - Best Single vs. Ensemble Models', fontsize=18, fontweight='bold')
ax8.set_xlim(0.85, 0.92); ax8.set_xlabel('Accuracy', fontsize=14); ax8.set_ylabel('Model Configuration', fontsize=14)
for p in ax8.patches:
    ax8.text(p.get_width(), p.get_y() + p.get_height() / 2, f' {p.get_width():.4f}', va='center')

# --- Row 4: The Champion Model Breakdown ---
ax9 = fig.add_subplot(gs[3, 0])
df_melted_f1 = pd.melt(df_champion_report, id_vars="Category", var_name="Metric", value_name="Score")
sns.barplot(x="Category", y="Score", hue="Metric", data=df_melted_f1, ax=ax9, palette="cubehelix")
ax9.set_title('Chart 9: Champion Performance Breakdown\n[Stack: e5, LR(t)] | Accuracy: 0.9040', fontsize=18, fontweight='bold')
ax9.set_ylim(0.7, 1.0); ax9.set_ylabel('Score', fontsize=14); ax9.set_xlabel('Scientific Field', fontsize=14)
ax9.legend(title='Metric')

ax10 = fig.add_subplot(gs[3, 1], polar=True)
categories = df_champion_report['Category'].tolist()
f1_scores = df_champion_report['F1-Score'].tolist()
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
f1_scores += f1_scores[:1] # Close the loop
angles += angles[:1]
ax10.plot(angles, f1_scores, 'o-', linewidth=2, color='crimson')
ax10.fill(angles, f1_scores, 'crimson', alpha=0.25)
ax10.set_thetagrids(np.degrees(angles[:-1]), categories)
ax10.set_title('Chart 10: Champion F1-Scores (Radar)', fontsize=18, fontweight='bold', y=1.15)
ax10.set_rlim(0.75, 1.0)

ax11 = fig.add_subplot(gs[3, 2])
# --- MODIFIED: Use the 'join=False' parameter and remove the warning-causing one ---
sns.pointplot(data=df_f1_melt, x='F1-Score', y='Model', hue='Category', ax=ax11, dodge=True, linestyles='none', palette='tab10')
ax11.set_title('Chart 11: Category F1-Scores of Top Models', fontsize=18, fontweight='bold')
ax11.set_xlabel('F1-Score', fontsize=14)
ax11.set_ylabel('Model Configuration', fontsize=14)
ax11.grid(axis='y')
ax11.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')


# --- Final Touches ---
plt.savefig('Definitive_Benchmark_Dashboard_v3.png', dpi=300)
plt.show()