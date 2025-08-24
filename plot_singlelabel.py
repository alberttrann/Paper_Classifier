# plot_generator.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# --- Plotting function for Plot 1 ---
def create_plot_1():
    """
    Generates and saves the plot for the single model benchmarks
    using hardcoded data for reliability and simplicity.
    """
    
    # --- Hardcoded Data from benchmark_results.md ---
    
    # 1. Overall Accuracy Data
    accuracy_data = {
        'BoW':        {'kNN': 0.3500, 'MNB': 0.8710, 'DT': 0.6130, 'KMeans': 0.3880},
        'TF-IDF':     {'kNN': 0.8010, 'MNB': 0.8670, 'DT': 0.6200, 'KMeans': 0.6990},
        'Embeddings': {'kNN': 0.8590, 'MNB': 0.8160, 'DT': 0.5110, 'KMeans': 0.7260}
    }
    
    # 2. Per-Class F1-Score Data
    f1_score_data = {
        # Model (Feature) -> {class: f1_score}
        'kNN (BoW)':        {'astro-ph': 0.48, 'cond-mat': 0.14, 'cs': 0.38, 'math': 0.40, 'physics': 0.16},
        'kNN (TF-IDF)':     {'astro-ph': 0.88, 'cond-mat': 0.81, 'cs': 0.81, 'math': 0.86, 'physics': 0.63},
        'kNN (Embeddings)': {'astro-ph': 0.94, 'cond-mat': 0.84, 'cs': 0.90, 'math': 0.91, 'physics': 0.69},
        'MNB (BoW)':        {'astro-ph': 0.93, 'cond-mat': 0.88, 'cs': 0.89, 'math': 0.94, 'physics': 0.71},
        'MNB (TF-IDF)':     {'astro-ph': 0.93, 'cond-mat': 0.87, 'cs': 0.88, 'math': 0.94, 'physics': 0.69},
        'MNB (Embeddings)': {'astro-ph': 0.92, 'cond-mat': 0.80, 'cs': 0.84, 'math': 0.89, 'physics': 0.60},
        'DT (BoW)':         {'astro-ph': 0.80, 'cond-mat': 0.64, 'cs': 0.65, 'math': 0.62, 'physics': 0.41},
        'DT (TF-IDF)':      {'astro-ph': 0.78, 'cond-mat': 0.60, 'cs': 0.66, 'math': 0.66, 'physics': 0.47},
        'DT (Embeddings)':  {'astro-ph': 0.64, 'cond-mat': 0.51, 'cs': 0.51, 'math': 0.57, 'physics': 0.31},
        'KMeans (BoW)':     {'astro-ph': 0.73, 'cond-mat': 0.37, 'cs': 0.06, 'math': 0.44, 'physics': 0.00},
        'KMeans (TF-IDF)':  {'astro-ph': 0.87, 'cond-mat': 0.72, 'cs': 0.71, 'math': 0.83, 'physics': 0.00},
        'KMeans (Emb)':     {'astro-ph': 0.86, 'cond-mat': 0.69, 'cs': 0.82, 'math': 0.88, 'physics': 0.00}
    }

    # --- Prepare data for plotting ---
    # 1. For the grouped bar chart (from dict to DataFrame)
    acc_df = pd.DataFrame(accuracy_data).T # Transpose to have features as rows
    # Reorder columns to a logical sequence
    acc_df = acc_df[['kNN', 'MNB', 'DT', 'KMeans']] 

    # 2. For the heatmap (from dict to DataFrame)
    f1_df = pd.DataFrame(f1_score_data).T # Transpose to have model/feature as rows
    # Ensure columns are in the correct order
    f1_df = f1_df[['astro-ph', 'cond-mat', 'cs', 'math', 'physics']] 
    
    # --- Create the plot ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [1, 1.2]})
    fig.suptitle('Plot 1: Single Model Benchmark Performance', fontsize=20, weight='bold')

    # Subplot 1: Grouped Bar Chart for Overall Accuracy
    acc_df.plot(kind='bar', ax=axes[0], colormap='viridis', width=0.8)
    axes[0].set_title('Overall Accuracy by Model and Feature Type', fontsize=14, weight='bold')
    axes[0].set_xlabel('Feature Representation', fontsize=12, weight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12, weight='bold')
    axes[0].tick_params(axis='x', rotation=0, labelsize=11)
    axes[0].grid(axis='x', linestyle='--')
    axes[0].legend(title='Models', fontsize=11)
    # Add accuracy values on top of bars
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%.4f', fontsize=8, padding=3, weight='bold')
    axes[0].set_ylim(0, 1.0)

    # Subplot 2: Heatmap for Per-Class F1-Scores
    sns.heatmap(f1_df, ax=axes[1], annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5,
                annot_kws={"weight": "bold"})
    axes[1].set_title('Per-Class F1-Score by Model and Feature Type', fontsize=14, weight='bold')
    axes[1].set_xlabel('Class Category', fontsize=12, weight='bold')
    axes[1].set_ylabel('Model (Feature Type)', fontsize=12, weight='bold')
    axes[1].tick_params(axis='y', rotation=0)

    # --- Final Touches and Save ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle and padding
    
    # Save the figure
    save_path = "plot_1_single_model_benchmarks.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot 1 saved successfully to '{save_path}'")
    plt.show()

def create_plot_2_and_3():
    """
    Generates and saves the plots for the ensemble model benchmarks
    using hardcoded data for reliability. This creates two separate plot files.
    """
    
    # --- Hardcoded Data from benchmark_results.md ---
    
    # Data for Plot 2: Embedding-Only Ensembles
    embedding_ensemble_data = {
        'MNB(emb) + kNN(emb)': 0.8340,
        'MNB(emb) + kNN(emb) + DT(emb)': 0.8280
    }
    
    # Data for Plot 3: Heterogeneous Ensembles
    heterogeneous_ensemble_data = {
        'Best Single Model (MNB(bow))': 0.8710, # For comparison
        'Best Embedding-Only Ensemble': 0.8340, # For comparison
        'MNB(bow) + kNN(emb)': 0.8580,
        'MNB(tfidf) + kNN(emb)': 0.8500,
        'MNB(bow) + kNN(emb) + DT(bow)': 0.8710,
        'MNB(tfidf) + kNN(emb) + DT(bow)': 0.8700,
        'MNB(tfidf) + kNN(emb) + DT(tfidf)': 0.8750,
        'MNB(bow) + kNN(emb) + DT(tfidf)': 0.8760, # The winner
    }
    
    # --- Create Plot 2: Embedding-Only Ensembles ---
    sns.set_theme(style="whitegrid")
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    
    names = list(embedding_ensemble_data.keys())
    accuracies = list(embedding_ensemble_data.values())
    
    # matplotlib's bar function returns a container object
    bars2 = ax2.bar(names, accuracies, color=sns.color_palette("rocket", 2))
    
    ax2.set_title('Plot 2: Performance of Embedding-Only Ensembles', fontsize=16, weight='bold')
    ax2.set_ylabel('Overall Accuracy', fontsize=12, weight='bold')
    ax2.set_xlabel('Ensemble Configuration', fontsize=12, weight='bold')
    ax2.set_ylim(0.82, 0.84) # Zoom in on the relevant accuracy range
    ax2.tick_params(axis='x', labelsize=10)
    ax2.grid(axis='x')
    
    # Add accuracy values on top of bars
    ax2.bar_label(bars2, fmt='%.4f', fontsize=11, weight='bold', padding=5)
    
    plt.tight_layout()
    save_path_2 = "plot_2_embedding_ensembles.png"
    plt.savefig(save_path_2, dpi=300, bbox_inches='tight')
    print(f"Plot 2 saved successfully to '{save_path_2}'")
    plt.show()

    # --- Create Plot 3: Heterogeneous Ensembles ---
    # Prepare data for Plot 3, sorting by performance for a clearer visual
    plot3_df = pd.DataFrame(list(heterogeneous_ensemble_data.items()), columns=['Configuration', 'Accuracy'])
    plot3_df = plot3_df.sort_values('Accuracy', ascending=False)
    
    sns.set_theme(style="whitegrid")
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    
    # Create a color palette where the winner stands out
    palette = []
    for conf in plot3_df['Configuration']:
        if 'MNB(bow) + kNN(emb) + DT(tfidf)' in conf:
            palette.append('salmon') # Winner color
        elif 'Best' in conf:
            palette.append('grey') # Baseline color
        else:
            palette.append('skyblue') # Other ensembles
            
    # Draw the plot using seaborn
    sns.barplot(x='Accuracy', y='Configuration', data=plot3_df, palette=palette, ax=ax3, orient='h')
    
    ax3.set_title('Plot 3: Heterogeneous Ensembles Outperform Baselines', fontsize=16, weight='bold')
    ax3.set_xlabel('Overall Accuracy', fontsize=12, weight='bold')
    ax3.set_ylabel('Ensemble Configuration', fontsize=12, weight='bold')
    ax3.set_xlim(0.8, 0.9) # Zoom in on the relevant accuracy range
    
    # Access the bar containers directly from the axes object
    for container in ax3.containers:
        ax3.bar_label(container, fmt='%.4f', fontsize=10, weight='bold', padding=5)
    # --- END OF FIX ---
    
    plt.tight_layout()
    save_path_3 = "plot_3_heterogeneous_ensembles.png"
    plt.savefig(save_path_3, dpi=300, bbox_inches='tight')
    print(f"Plot 3 saved successfully to '{save_path_3}'")
    plt.show()

def create_plot_4():
    """
    Generates and saves the plot for the stacking ensemble benchmarks (Section 4)
    using hardcoded data.
    """
    
    # --- Hardcoded Data from benchmark_results.md (Section 4) ---
    stacking_data = {
        'Stack 1: [MNB(b)+kNN(e)+DT(t)] + LR(b)': 0.8870,
        'Stack 2: [MNB(b)+kNN(e)+DT(t)] + LR(t)': 0.8950,
        'Stack 3: [MNB(b)+kNN(e)+DT(t)] + LR(e)': 0.8870,
        'Stack 4: [MNB(t)+kNN(e)+DT(t)] + LR(b)': 0.8820,
        'Stack 5: [MNB(t)+kNN(e)+DT(t)] + LR(t)': 0.8980,
        'Stack 6: [MNB(t)+kNN(e)+DT(t)] + LR(e)': 0.8930,
        'Stack 7: [MNB(b)+kNN(e)+DT(t)] + DT(t)': 0.8570,
        'Stack 8: [MNB(b)+kNN(e)] + DT(t)': 0.8700,
        'Stack 9: [MNB(b)+kNN(e)] + LR(b)': 0.8850,
        'Stack 10: [MNB(b)+kNN(e)] + LR(t)': 0.8910,
        'Stack 11: [MNB(b)+kNN(e)] + LR(e)': 0.8870,
    }

    # --- Prepare data for plotting by parsing the configuration strings ---
    parsed_data = []
    for config, accuracy in stacking_data.items():
        base_models_str, meta_learner_str = config.split('] + ')
        base_models_str = base_models_str.split(': [')[1]
        
        parsed_data.append({
            'Base Models': base_models_str,
            'Meta-Learner': meta_learner_str,
            'Accuracy': accuracy
        })
        
    df = pd.DataFrame(parsed_data)
    
    # Pivot the DataFrame to create the structure needed for a grouped bar chart
    pivot_df = df.pivot(index='Base Models', columns='Meta-Learner', values='Accuracy')
    # Reorder for a more logical presentation
    pivot_df = pivot_df.reindex([
        'MNB(b)+kNN(e)+DT(t)', 
        'MNB(t)+kNN(e)+DT(t)', 
        'MNB(b)+kNN(e)'
    ])
    # Reorder meta-learner columns
    pivot_df = pivot_df[['LR(b)', 'LR(t)', 'LR(e)', 'DT(t)']]


    # --- Create the plot ---
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(16, 9))
    
    pivot_df.plot(kind='bar', ax=ax, colormap='Spectral', width=0.8)
    
    ax.set_title('Plot 4: Stacking Ensemble Performance by Configuration', fontsize=20, weight='bold')
    ax.set_xlabel('Base Model Combination', fontsize=14, weight='bold')
    ax.set_ylabel('Overall Accuracy', fontsize=14, weight='bold')
    ax.set_ylim(0.85, 0.91) # Zoom in on the top-tier performance
    ax.tick_params(axis='x', rotation=15, labelsize=12)
    ax.grid(axis='y', linestyle='--')
    
    # Customize legend
    ax.legend(title='Meta-Learner (Features)', title_fontsize='13', fontsize='11', loc='upper right')
    
    # Add accuracy values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', fontsize=9, padding=3, weight='bold', rotation=90)
    
    # --- Final Touches and Save ---
    plt.tight_layout()
    save_path = "plot_4_stacking_ensembles.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot 4 saved successfully to '{save_path}'")
    plt.show()

def create_plots_5_and_6():
    """
    Generates and saves the plots for the Ultimate Benchmark comparison
    between SciBERT and e5-base at two data scales.
    """
    
    # --- Hardcoded Data from benchmark_results.md (Sections 5 & 6) ---
    
    # Data for 1000 samples/category
    data_1k = {
        'Model Configuration': [
            'LR on Combined', 
            'Soft Voting', 
            'Pure Stacking', 
            'Stacking+GNB', 
            'Confidence-Gated'
        ],
        'SciBERT':  [0.8690, 0.8850, 0.8930, 0.8840, 0.8590],
        'e5-base':  [0.8850, 0.8920, 0.9010, 0.8900, 0.8590]
    }
    df_1k = pd.DataFrame(data_1k)
    
    # Data for 2000 samples/category
    data_2k = {
        'Model Configuration': [
            'LR on Combined', 
            'Soft Voting', 
            'Pure Stacking', 
            'Stacking+GNB', 
            'Confidence-Gated'
        ],
        'SciBERT':  [0.8570, 0.8850, 0.8850, 0.8790, 0.8785],
        'e5-base':  [0.8820, 0.8870, 0.8895, 0.8770, 0.8800]
    }
    df_2k = pd.DataFrame(data_2k)

    # --- Create Plot 5: 1000 Samples/Category ---
    sns.set_theme(style="whitegrid")
    fig5, ax5 = plt.subplots(figsize=(16, 9))
    
    # Melt the dataframe to prepare for grouped bar plotting
    df_1k_melted = df_1k.melt(id_vars='Model Configuration', var_name='Embedding Model', value_name='Accuracy')
    
    sns.barplot(x='Model Configuration', y='Accuracy', hue='Embedding Model', data=df_1k_melted, ax=ax5, palette='cividis')

    ax5.set_title('Plot 5: In Search of Models Better than Stacking (1000 Samples/Category)', fontsize=18, weight='bold')
    ax5.set_xlabel('Ensemble/Model Type', fontsize=14, weight='bold')
    ax5.set_ylabel('Overall Accuracy', fontsize=14, weight='bold')
    ax5.set_ylim(0.84, 0.92) # Zoom in
    ax5.tick_params(axis='x', rotation=10, labelsize=12)
    ax5.legend(title='Embedding Model', title_fontsize='13', fontsize='11')
    
    # Add accuracy values on top of bars
    for container in ax5.containers:
        ax5.bar_label(container, fmt='%.4f', fontsize=10, padding=3, weight='bold')

    plt.tight_layout()
    save_path_5 = "plot_5_experiments_comparison_1k.png"
    plt.savefig(save_path_5, dpi=300, bbox_inches='tight')
    print(f"Plot 5 saved successfully to '{save_path_5}'")
    plt.show()

    # --- Create Plot 6: 2000 Samples/Category ---
    sns.set_theme(style="whitegrid")
    fig6, ax6 = plt.subplots(figsize=(16, 9))
    
    # Melt the dataframe
    df_2k_melted = df_2k.melt(id_vars='Model Configuration', var_name='Embedding Model', value_name='Accuracy')

    sns.barplot(x='Model Configuration', y='Accuracy', hue='Embedding Model', data=df_2k_melted, ax=ax6, palette='cividis')

    ax6.set_title('Plot 6: In Search of Models Better than Stacking (2000 Samples/Category)', fontsize=18, weight='bold')
    ax6.set_xlabel('Ensemble/Model Type', fontsize=14, weight='bold')
    ax6.set_ylabel('Overall Accuracy', fontsize=14, weight='bold')
    ax6.set_ylim(0.84, 0.92) # Use same y-axis limit for fair comparison
    ax6.tick_params(axis='x', rotation=10, labelsize=12)
    ax6.legend(title='Embedding Model', title_fontsize='13', fontsize='11')

    # Add accuracy values on top of bars
    for container in ax6.containers:
        ax6.bar_label(container, fmt='%.4f', fontsize=10, padding=3, weight='bold')

    plt.tight_layout()
    save_path_6 = "plot_6_experiments_comparison_2k.png"
    plt.savefig(save_path_6, dpi=300, bbox_inches='tight')
    print(f"Plot 6 saved successfully to '{save_path_6}'")
    plt.show()

def create_plot_6_1():
    """
    Generates and saves a plot to visualize the effect of data scaling
    (1k vs 2k samples/category) across all advanced model configurations
    for both SciBERT and e5-base.
    """
    
    # --- Hardcoded Data from benchmark_results.md (Sections 5 & 6) ---
    
    # Combine all data into a single structure for easier plotting
    data = {
        'Model Configuration': [
            'LR on Combined', 'Soft Voting', 'Pure Stacking', 'Stacking+GNB', 'Confidence-Gated',
            'LR on Combined', 'Soft Voting', 'Pure Stacking', 'Stacking+GNB', 'Confidence-Gated',
            'LR on Combined', 'Soft Voting', 'Pure Stacking', 'Stacking+GNB', 'Confidence-Gated',
            'LR on Combined', 'Soft Voting', 'Pure Stacking', 'Stacking+GNB', 'Confidence-Gated',
        ],
        'Accuracy': [
            # SciBERT 1k
            0.8690, 0.8850, 0.8930, 0.8840, 0.8590,
            # e5-base 1k
            0.8850, 0.8920, 0.9010, 0.8900, 0.8590,
            # SciBERT 2k
            0.8570, 0.8850, 0.8850, 0.8790, 0.8785,
            # e5-base 2k
            0.8820, 0.8870, 0.8895, 0.8770, 0.8800,
        ],
        'Data Scale': 
            ['1k Samples/Cat'] * 10 + ['2k Samples/Cat'] * 10,
        'Embedding Model': 
            ['SciBERT'] * 5 + ['e5-base'] * 5 + ['SciBERT'] * 5 + ['e5-base'] * 5
    }
    
    df = pd.DataFrame(data)
    
    # --- Create the Plot ---
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(18, 10))

    # We can create a combined category for the hue to get 4 distinct bars
    df['Hue'] = df['Embedding Model'] + ' (' + df['Data Scale'] + ')'
    
    sns.barplot(x='Model Configuration', y='Accuracy', hue='Hue', data=df, ax=ax,
                palette={'SciBERT (1k Samples/Cat)': 'lightblue', 'SciBERT (2k Samples/Cat)': 'blue',
                         'e5-base (1k Samples/Cat)': 'lightcoral', 'e5-base (2k Samples/Cat)': 'red'})

    ax.set_title('Plot 6.1: Effect of Data Scaling on Model Performance', fontsize=22, weight='bold')
    ax.set_xlabel('Ensemble/Model Type', fontsize=15, weight='bold')
    ax.set_ylabel('Overall Accuracy', fontsize=15, weight='bold')
    ax.set_ylim(0.84, 0.92) # Zoom in on the relevant accuracy range
    ax.tick_params(axis='x', rotation=10, labelsize=13)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(title='Experiment', title_fontsize='14', fontsize='12')

    # Add accuracy values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', fontsize=9, padding=3, weight='bold', rotation=45)

    plt.tight_layout()
    save_path = "plot_6_1_data_scaling_effect.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot 6.1 saved successfully to '{save_path}'")
    plt.show()

def create_plot_7():
    """
    Generates and saves the plot for the Champion Pipeline benchmarks (Section 7),
    comparing SciBERT and e5-base results side-by-side.
    """
    
    # --- Hardcoded Data from benchmark_results.md (Section 7) ---
    
    # Data for SciBERT @ 1000 samples/category
    scibert_1k_data = {
        'LR(TFIDF)': 0.8940, 'LR(BoW)': 0.8850, 'LR(Emb)': 0.8780,
        'XGB(TFIDF)': 0.8940, 'XGB(BoW)': 0.8860, 'XGB(Emb)': 0.8940,
        'GNB(TFIDF)': 0.7810, 'GNB(BoW)': 0.7640, 'GNB(Emb)': 0.8630,
    }

    # Data for e5-base @ 1000 samples/category
    e5_base_1k_data = {
        'LR(TFIDF)': 0.9040, 'LR(BoW)': 0.8840, 'LR(Emb)': 0.9020,
        'XGB(TFIDF)': 0.9020, 'XGB(BoW)': 0.9030, 'XGB(Emb)': 0.9030,
        'GNB(TFIDF)': 0.7810, 'GNB(BoW)': 0.7640, 'GNB(Emb)': 0.8700,
    }
    
    # We will use the 1k/category data as it produced the highest scores.
    # The logic could be easily adapted for the 2k data if needed.
    
    # --- Prepare data for plotting by creating a structured DataFrame ---
    plot_data = []
    for model_config, accuracy in scibert_1k_data.items():
        meta_learner, features = model_config.replace(')', '').split('(')
        plot_data.append({
            'Embedding Model': 'SciBERT',
            'Meta-Learner': meta_learner,
            'Meta-Features': features,
            'Accuracy': accuracy
        })
        
    for model_config, accuracy in e5_base_1k_data.items():
        meta_learner, features = model_config.replace(')', '').split('(')
        plot_data.append({
            'Embedding Model': 'e5-base',
            'Meta-Learner': meta_learner,
            'Meta-Features': features,
            'Accuracy': accuracy
        })

    df = pd.DataFrame(plot_data)

    # --- Create the plot with two subplots ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(22, 10), sharey=True)
    fig.suptitle('Plot 7: Champion Stacking Pipeline Performance by Meta-Learner (1000 Samples/Category)', fontsize=20, weight='bold')

    # Subplot 1: SciBERT Results
    scibert_df = df[df['Embedding Model'] == 'SciBERT']
    sns.barplot(ax=axes[0], x='Meta-Learner', y='Accuracy', hue='Meta-Features', data=scibert_df, palette='plasma')
    axes[0].set_title('Base kNN using SciBERT Embeddings', fontsize=18, weight='bold')
    axes[0].set_xlabel('Meta-Learner Algorithm', fontsize=14, weight='bold')
    axes[0].set_ylabel('Overall Accuracy', fontsize=14, weight='bold')
    axes[0].set_ylim(0.75, 0.92)
    axes[0].legend(title='Meta-Features', title_fontsize='13', fontsize='11')
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%.4f', fontsize=10, padding=3, weight='bold')

    # Subplot 2: e5-base Results
    e5_df = df[df['Embedding Model'] == 'e5-base']
    sns.barplot(ax=axes[1], x='Meta-Learner', y='Accuracy', hue='Meta-Features', data=e5_df, palette='viridis')
    axes[1].set_title('Base kNN using e5-base Embeddings', fontsize=18, weight='bold')
    axes[1].set_xlabel('Meta-Learner Algorithm', fontsize=14, weight='bold')
    axes[1].set_ylabel('') # Hide redundant y-label
    axes[1].legend(title='Meta-Features', title_fontsize='13', fontsize='11')
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%.4f', fontsize=10, padding=3, weight='bold')

    # --- Final Touches and Save ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = "plot_7_champion_pipeline_1k.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot 7 saved successfully to '{save_path}'")
    plt.show()    

def create_plots_7_1():
    """
    Generates and saves two plots:
    - Plot 7.1: The Champion Pipeline benchmark for the 2000 samples/category data.
    - Plot 7.2: The effect of data scale on key model architectures.
    """
    
    # --- Hardcoded Data from benchmark_results.md (Section 7 and others) ---

    # Data for Plot 7.1: Champion Pipeline @ 2000 samples/category
    scibert_2k_data = {
        'LR(TFIDF)': 0.8900, 'LR(BoW)': 0.8865, 'LR(Emb)': 0.8755,
        'XGB(TFIDF)': 0.8885, 'XGB(BoW)': 0.8960, 'XGB(Emb)': 0.8880,
        'GNB(TFIDF)': 0.7775, 'GNB(BoW)': 0.7420, 'GNB(Emb)': 0.8490,
    }
    e5_base_2k_data = {
        'LR(TFIDF)': 0.8920, 'LR(BoW)': 0.8825, 'LR(Emb)': 0.8920,
        'XGB(TFIDF)': 0.8940, 'XGB(BoW)': 0.8915, 'XGB(Emb)': 0.8975,
        'GNB(TFIDF)': 0.7775, 'GNB(BoW)': 0.7425, 'GNB(Emb)': 0.8545,
    }
    
    # Data for Plot 7.2: Data Scaling Effect
    # We'll compare our best single model, best voting, and best stacking model
    # using the superior e5-base embedding results.
    scaling_data = {
        'Model': [
            'Best Single Model (LR-TFIDF)',
            'Soft Voting Ensemble',
            'Pure Stacking + LR',
            'Champion Stack (XGB-Emb)' # Picking XGB(Emb) as the 2k winner
        ],
        '1000 Samples/Category': [
            0.8700, # From run_single_model_benchmark.py (1k)
            0.8920, # From run_ultimate_benchmark_e5.py (1k)
            0.9010, # From run_ultimate_benchmark_e5.py (1k)
            0.9030, # From champion_e5.py (1k)
        ],
        '2000 Samples/Category': [
            0.8710, # From run_single_model_benchmark.py (2k)
            0.8870, # From run_ultimate_benchmark_e5.py (2k)
            0.8895, # From run_ultimate_benchmark_e5.py (2k)
            0.8975, # From champion_e5.py (2k)
        ]
    }

    # --- Create Plot 7.1: Champion Pipeline Performance (2000 Samples/Category) ---
    df_2k_plot_data = []
    for model_config, accuracy in scibert_2k_data.items():
        meta_learner, features = model_config.replace(')', '').split('(')
        df_2k_plot_data.append({'Embedding Model': 'SciBERT', 'Meta-Learner': meta_learner, 'Meta-Features': features, 'Accuracy': accuracy})
    for model_config, accuracy in e5_base_2k_data.items():
        meta_learner, features = model_config.replace(')', '').split('(')
        df_2k_plot_data.append({'Embedding Model': 'e5-base', 'Meta-Learner': meta_learner, 'Meta-Features': features, 'Accuracy': accuracy})
    df_2k = pd.DataFrame(df_2k_plot_data)

    sns.set_theme(style="whitegrid")
    fig7_1, axes7_1 = plt.subplots(1, 2, figsize=(22, 10), sharey=True)
    fig7_1.suptitle('Plot 7.1: Champion Stacking Pipeline Performance (2000 Samples/Category)', fontsize=24, weight='bold')

    # Subplot 1: SciBERT Results (2k)
    scibert_df_2k = df_2k[df_2k['Embedding Model'] == 'SciBERT']
    sns.barplot(ax=axes7_1[0], x='Meta-Learner', y='Accuracy', hue='Meta-Features', data=scibert_df_2k, palette='plasma')
    axes7_1[0].set_title('Base kNN using SciBERT Embeddings', fontsize=18, weight='bold')
    axes7_1[0].set_xlabel('Meta-Learner Algorithm', fontsize=14, weight='bold')
    axes7_1[0].set_ylabel('Overall Accuracy', fontsize=14, weight='bold')
    axes7_1[0].set_ylim(0.72, 0.92)
    axes7_1[0].legend(title='Meta-Features', title_fontsize='13', fontsize='11')
    for container in axes7_1[0].containers:
        axes7_1[0].bar_label(container, fmt='%.4f', fontsize=10, padding=3, weight='bold')

    # Subplot 2: e5-base Results (2k)
    e5_df_2k = df_2k[df_2k['Embedding Model'] == 'e5-base']
    sns.barplot(ax=axes7_1[1], x='Meta-Learner', y='Accuracy', hue='Meta-Features', data=e5_df_2k, palette='viridis')
    axes7_1[1].set_title('Base kNN using e5-base Embeddings', fontsize=18, weight='bold')
    axes7_1[1].set_xlabel('Meta-Learner Algorithm', fontsize=14, weight='bold')
    axes7_1[1].set_ylabel('') # Hide redundant y-label
    axes7_1[1].legend(title='Meta-Features', title_fontsize='13', fontsize='11')
    for container in axes7_1[1].containers:
        axes7_1[1].bar_label(container, fmt='%.4f', fontsize=10, padding=3, weight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path_7_1 = "plot_7_1_champion_pipeline_2k.png"
    plt.savefig(save_path_7_1, dpi=300, bbox_inches='tight')
    print(f"Plot 7.1 saved successfully to '{save_path_7_1}'")
    plt.show()

def create_final_plot_7_2():
    """
    Generates the final summary plot for the Champion Pipeline (Section 7),
    visualizing the effect of data scaling and embedding model choice
    in a layout similar to the provided example.
    """
    
    # --- Hardcoded Data from benchmark_results.md (Section 7) ---
    
    # Data for 1000 samples/category
    scibert_1k_data = {
        'LR(TFIDF)': 0.8940, 'LR(BoW)': 0.8850, 'LR(Emb)': 0.8780,
        'XGB(TFIDF)': 0.8940, 'XGB(BoW)': 0.8860, 'XGB(Emb)': 0.8940,
        'GNB(TFIDF)': 0.7810, 'GNB(BoW)': 0.7640, 'GNB(Emb)': 0.8630,
    }
    e5_base_1k_data = {
        'LR(TFIDF)': 0.9040, 'LR(BoW)': 0.8840, 'LR(Emb)': 0.9020,
        'XGB(TFIDF)': 0.9020, 'XGB(BoW)': 0.9030, 'XGB(Emb)': 0.9030,
        'GNB(TFIDF)': 0.7810, 'GNB(BoW)': 0.7640, 'GNB(Emb)': 0.8700,
    }
    
    # Data for 2000 samples/category
    scibert_2k_data = {
        'LR(TFIDF)': 0.8900, 'LR(BoW)': 0.8865, 'LR(Emb)': 0.8755,
        'XGB(TFIDF)': 0.8885, 'XGB(BoW)': 0.8960, 'XGB(Emb)': 0.8880,
        'GNB(TFIDF)': 0.7775, 'GNB(BoW)': 0.7420, 'GNB(Emb)': 0.8490,
    }
    e5_base_2k_data = {
        'LR(TFIDF)': 0.8920, 'LR(BoW)': 0.8825, 'LR(Emb)': 0.8920,
        'XGB(TFIDF)': 0.8940, 'XGB(BoW)': 0.8915, 'XGB(Emb)': 0.8975,
        'GNB(TFIDF)': 0.7775, 'GNB(BoW)': 0.7425, 'GNB(Emb)': 0.8545,
    }

    # --- Prepare data for plotting by creating a structured DataFrame ---
    plot_data = []
    
    # Function to populate the list
    def populate_data(data_dict, emb_model, scale):
        for config, accuracy in data_dict.items():
            meta_learner, features = config.replace(')', '').split('(')
            plot_data.append({
                'Experiment': f"{emb_model} ({scale})",
                'Meta-Learner Config': config,
                'Accuracy': accuracy
            })

    populate_data(scibert_1k_data, 'SciBERT', '1k Samples/Cat')
    populate_data(e5_base_1k_data, 'e5-base', '1k Samples/Cat')
    populate_data(scibert_2k_data, 'SciBERT', '2k Samples/Cat')
    populate_data(e5_base_2k_data, 'e5-base', '2k Samples/Cat')
        
    df = pd.DataFrame(plot_data)

    # --- Create the plot ---
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Create the grouped bar plot using seaborn
    sns.barplot(
        data=df,
        x='Meta-Learner Config',
        y='Accuracy',
        hue='Experiment',
        palette={'SciBERT (1k Samples/Cat)': 'lightblue', 'e5-base (1k Samples/Cat)': 'lightcoral',
                 'SciBERT (2k Samples/Cat)': 'blue', 'e5-base (2k Samples/Cat)': 'red'},
        ax=ax
    )
    
    ax.set_title('Plot 7.2: Effect of Data Scale & Embedding Model on Champion Pipeline', fontsize=24, weight='bold')
    ax.set_xlabel('Meta-Learner Configuration', fontsize=16, weight='bold')
    ax.set_ylabel('Overall Accuracy', fontsize=16, weight='bold')
    ax.set_ylim(0.7, 0.92) # Set y-limit to start from a reasonable baseline
    ax.tick_params(axis='x', rotation=15, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(title='Experiment', title_fontsize='14', fontsize='12')
    
    # Add accuracy values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', fontsize=9, padding=3, weight='bold', rotation=45)
        
    # --- Final Touches and Save ---
    plt.tight_layout()
    save_path = "plot_7_2_champion_scaling_effect.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot 7.2 saved successfully to '{save_path}'")
    plt.show()


def create_plot_8_detailed_benchmark():
    """
    Generates a detailed 3-part plot for Section 8, analyzing the performance
    of LogisticRegression and XGBoost across different features and data scales.
    """
    
    # --- Hardcoded Data from benchmark_results.md (Section 8) ---
    
    # Data for 1000 samples/category
    data_1k = {
        'LR(BoW)': 0.8580, 'LR(TFIDF)': 0.8700, 'LR(Emb)': 0.8560,
        'XGB(BoW)': 0.8230, 'XGB(TFIDF)': 0.8270, 'XGB(Emb)': 0.8380,
    }

    # Data for 2000 samples/category
    data_2k = {
        'LR(BoW)': 0.8515, 'LR(TFIDF)': 0.8710, 'LR(Emb)': 0.8400,
        'XGB(BoW)': 0.8345, 'XGB(TFIDF)': 0.8360, 'XGB(Emb)': 0.8350,
    }
    
    # --- Prepare data for plotting by creating a structured DataFrame ---
    plot_data = []
    def populate_data(data_dict, scale):
        for config, accuracy in data_dict.items():
            model, features = config.replace(')', '').split('(')
            plot_data.append({
                'Scale': scale,
                'Model': model,
                'Features': features,
                'Accuracy': accuracy
            })

    populate_data(data_1k, '1k Samples/Cat')
    populate_data(data_2k, '2k Samples/Cat')
        
    df = pd.DataFrame(plot_data)

    # --- Create the 3-part plot ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle('Plot 8: Performance Benchmark of Single Model Challengers (LR vs. XGBoost)', fontsize=24, weight='bold')

    # --- Subplot 1: 1000 Samples/Category Benchmark ---
    df_1k = df[df['Scale'] == '1k Samples/Cat']
    sns.barplot(ax=axes[0], data=df_1k, x='Model', y='Accuracy', hue='Features', palette='rocket')
    axes[0].set_title('Performance at 1000 Samples/Category', fontsize=16, weight='bold')
    axes[0].set_xlabel('Model', fontsize=12, weight='bold')
    axes[0].set_ylabel('Overall Accuracy', fontsize=12, weight='bold')
    axes[0].set_ylim(0.80, 0.90)
    axes[0].legend(title='Features')
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%.4f', fontsize=9, padding=3)

    # --- Subplot 2: 2000 Samples/Category Benchmark ---
    df_2k = df[df['Scale'] == '2k Samples/Cat']
    sns.barplot(ax=axes[1], data=df_2k, x='Model', y='Accuracy', hue='Features', palette='mako')
    axes[1].set_title('Performance at 2000 Samples/Category', fontsize=16, weight='bold')
    axes[1].set_xlabel('Model', fontsize=12, weight='bold')
    axes[1].set_ylabel('') # Hide redundant y-label
    axes[1].set_ylim(0.80, 0.90)
    axes[1].legend(title='Features')
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%.4f', fontsize=9, padding=3)
        
    # --- Subplot 3: Effect of Data Scale ---
    # We will plot the best configuration for each model (LR(TFIDF) and XGB(Emb))
    scaling_df_data = {
        'Model': ['LR(TFIDF)', 'XGB(Emb)'],
        '1k Samples/Cat': [data_1k['LR(TFIDF)'], data_1k['XGB(Emb)']],
        '2k Samples/Cat': [data_2k['LR(TFIDF)'], data_2k['XGB(Emb)']]
    }
    scaling_df = pd.DataFrame(scaling_df_data).melt(id_vars='Model', var_name='Dataset Size', value_name='Accuracy')
    
    sns.lineplot(ax=axes[2], data=scaling_df, x='Dataset Size', y='Accuracy', hue='Model', 
                 style='Model', markers=True, dashes=False, lw=3, markersize=10)
    axes[2].set_title('Effect of Data Scale on Best Configs', fontsize=16, weight='bold')
    axes[2].set_xlabel('Dataset Size', fontsize=12, weight='bold')
    axes[2].set_ylabel('Overall Accuracy', fontsize=12, weight='bold')
    axes[2].set_ylim(0.80, 0.90)
    axes[2].grid(linestyle='--', which='both')
    
    # Annotate points
    for i in range(2): # For LR and XGB
        axes[2].text(0, scaling_df.iloc[i, 2] + 0.001, f"{scaling_df.iloc[i, 2]:.4f}", va='bottom', ha='center', weight='bold')
        axes[2].text(1, scaling_df.iloc[i+2, 2] + 0.001, f"{scaling_df.iloc[i+2, 2]:.4f}", va='bottom', ha='center', weight='bold')

    # --- Final Touches and Save ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = "plot_8_detailed_challenger_benchmark.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot 8 saved successfully to '{save_path}'")
    plt.show()

def create_summary_plots():
    """
    Generates the final two summary plots for the entire project:
    - Plot 8: A "stairway to performance" chart showing model improvement.
    - Plot 9: A heatmap summarizing the peak performance of each model/feature combination.
    """
    
    # --- Hardcoded Data from all benchmark_results.md sections ---
    
    # Data for Plot 8: The "Stairway to Performance"
    stairway_data = {
        '1. Best Single Model\n(MNB on BoW)': 0.8710,
        '2. Best Homogeneous Ensemble\n(MNB+kNN on Embeddings)': 0.8340,
        '3. Best Heterogeneous Voting\n(MNB(b)+kNN(e)+DT(t))': 0.8760,
        '4. Best Ultimate Soft Voting\n(e5-base @ 1k)': 0.8920,
        '5. CHAMPION: Best Stacking\n(e5-base @ 1k, LR(TFIDF) Meta)': 0.9040
    }
    
    # Data for Plot 9: The Summary Heatmap
    heatmap_data = {
        # Rows: Models, Columns: Features
        'kNN':      {'BoW': 0.3500, 'TF-IDF': 0.8010, 'Embeddings': 0.8590},
        'MNB':      {'BoW': 0.8710, 'TF-IDF': 0.8670, 'Embeddings': 0.8160},
        'DT':       {'BoW': 0.6130, 'TF-IDF': 0.6200, 'Embeddings': 0.5110},
        'LR':       {'BoW': 0.8515, 'TF-IDF': 0.8710, 'Embeddings': 0.8400},
        'XGBoost':  {'BoW': 0.8345, 'TF-IDF': 0.8360, 'Embeddings': 0.8350},
        'Voting':   {'BoW': None, 'TF-IDF': None, 'Embeddings (Homogeneous)': 0.8340, 'Heterogeneous': 0.8760},
        'Stacking': {'BoW': None, 'TF-IDF': None, 'Embeddings (Homogeneous)': None, 'Heterogeneous': 0.9040}
    }

    # --- Create Plot 8: The "Stairway to Performance" ---
    stairway_df = pd.DataFrame(list(stairway_data.items()), columns=['Stage', 'Accuracy'])
    
    sns.set_theme(style="whitegrid")
    fig8, ax8 = plt.subplots(figsize=(12, 8))
    
    # Draw the plot using seaborn
    sns.barplot(x='Accuracy', y='Stage', data=stairway_df, palette='magma', ax=ax8, orient='h', errorbar=None)

    ax8.set_title('Plot 8: The Stairway to Performance - Model Evolution', fontsize=18, weight='bold')
    ax8.set_xlabel('Overall Accuracy', fontsize=12, weight='bold')
    ax8.set_ylabel('Experimental Stage', fontsize=12, weight='bold')
    ax8.set_xlim(0.82, 0.92) # Zoom in to highlight the improvements
    
    # --- THIS IS THE FIX ---
    # Access the bar containers directly from the axes object to add labels
    for container in ax8.containers:
        ax8.bar_label(container, fmt='%.4f', fontsize=11, weight='bold', padding=5)
    # --- END OF FIX ---
    
    # Add a line to emphasize the champion
    champion_acc = stairway_df['Accuracy'].max()
    ax8.axvline(x=champion_acc, color='gold', linestyle='--', linewidth=2.5, label=f'Champion Model ({champion_acc:.4f})')
    ax8.legend()
    
    plt.tight_layout()
    save_path_8 = "plot_8_performance_stairway.png"
    plt.savefig(save_path_8, dpi=300, bbox_inches='tight')
    print(f"Plot 8 saved successfully to '{save_path_8}'")
    plt.show()

    # --- Create Plot 9: Comprehensive Summary Heatmap ---
    # Convert dict to DataFrame, handling the different feature names for ensembles
    heatmap_df = pd.DataFrame(heatmap_data).T.fillna(0) # Transpose and fill NaNs
    # Reorder columns for logical flow
    heatmap_df = heatmap_df[['BoW', 'TF-IDF', 'Embeddings', 'Embeddings (Homogeneous)', 'Heterogeneous']]
    # Reorder rows
    heatmap_df = heatmap_df.reindex(['Stacking', 'Voting', 'MNB', 'LR', 'kNN', 'XGBoost', 'DT'])
    
    sns.set_theme(style="white")
    fig9, ax9 = plt.subplots(figsize=(14, 9))

    sns.heatmap(
        heatmap_df, 
        annot=True, 
        fmt=".4f", 
        cmap="BuPu", 
        linewidths=1, 
        linecolor='white',
        ax=ax9,
        annot_kws={"weight": "bold", "size": 12},
        cbar_kws={'label': 'Peak Accuracy'}
    )
    
    ax9.set_title('Plot 9: Peak Performance Summary - Models vs. Feature Types', fontsize=18, weight='bold')
    ax9.set_xlabel('Feature Representation Type', fontsize=12, weight='bold')
    ax9.set_ylabel('Model Architecture', fontsize=12, weight='bold')
    ax9.tick_params(axis='x', rotation=20)
    ax9.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    save_path_9 = "plot_9_final_summary_heatmap.png"
    plt.savefig(save_path_9, dpi=300, bbox_inches='tight')
    print(f"Plot 9 saved successfully to '{save_path_9}'")
    plt.show()

def create_final_summary_plots():
    """
    Generates the final 3-part summary figure, including the Champion Performance
    Breakdown, a Radar Chart, and a comparative Scatter Plot of top models.
    """
    
    # --- Hardcoded Data from benchmark_results.md ---
    # This data is meticulously collected from the best-performing runs.
    
    # Data for Plot 9 & 10: The Champion Model
    # Champion: Stacking [MNB(t)+kNN(e)+DT(t)] + LR(t) with e5-base @ 1000 samples/cat
    champion_accuracy = 0.9040
    champion_data = {
        'Class': ['astro-ph', 'cond-mat', 'cs', 'math', 'physics'],
        'Precision': [0.97, 0.89, 0.93, 0.92, 0.81],
        'Recall':    [0.94, 0.91, 0.90, 0.98, 0.79],
        'F1-Score':  [0.96, 0.90, 0.91, 0.95, 0.80]
    }
    df_champion = pd.DataFrame(champion_data)

    # Data for Plot 11: Comparison of Top Models
    # We'll compare the per-class F1-scores of key milestone models
    top_models_f1_data = {
        'Single MNB(BoW)': {'astro-ph': 0.93, 'cond-mat': 0.88, 'cs': 0.89, 'math': 0.94, 'physics': 0.71},
        'Single LR(TFIDF)': {'astro-ph': 0.94, 'cond-mat': 0.88, 'cs': 0.88, 'math': 0.93, 'physics': 0.73},
        'Voting Hetero.': {'astro-ph': 0.93, 'cond-mat': 0.87, 'cs': 0.89, 'math': 0.94, 'physics': 0.74},
        'Stacking [e5, LR(t)]': {'astro-ph': 0.96, 'cond-mat': 0.90, 'cs': 0.91, 'math': 0.95, 'physics': 0.80},
        'Stacking [e5, XGB(e)]': {'astro-ph': 0.96, 'cond-mat': 0.90, 'cs': 0.91, 'math': 0.95, 'physics': 0.79} # Using XGB(Emb) as a top contender
    }
    
    # Prepare data for scatter plot
    plot11_data = []
    for model_config, f1_scores in top_models_f1_data.items():
        for category, f1_score in f1_scores.items():
            plot11_data.append({
                'Model Configuration': model_config,
                'Category': category,
                'F1-Score': f1_score
            })
    df_plot11 = pd.DataFrame(plot11_data)
    
    
    # --- Create the 3-part plot ---
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(24, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1.2])

    # --- Plot 9 (Left): Champion Performance Breakdown ---
    ax9 = fig.add_subplot(gs[0])
    df_champion_melted = df_champion.melt(id_vars='Class', var_name='Metric', value_name='Score')
    sns.barplot(data=df_champion_melted, x='Class', y='Score', hue='Metric', ax=ax9, palette='PuBuGn_d')
    ax9.set_title(f'Chart 9: Champion Performance Breakdown\n[Stack: e5, LR(t)] | Accuracy: {champion_accuracy:.4f}', fontsize=16, weight='bold')
    ax9.set_xlabel('Scientific Field', fontsize=12, weight='bold')
    ax9.set_ylabel('Score', fontsize=12, weight='bold')
    ax9.set_ylim(0.70, 1.01)
    ax9.legend(title='Metric', title_fontsize='13', fontsize='11')
    
    # --- Plot 10 (Middle): Champion F1-Scores (Radar) ---
    ax10 = fig.add_subplot(gs[1], polar=True)
    
    # Set up the radar chart
    labels = df_champion['Class'].values
    stats = df_champion['F1-Score'].values
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    # Make the plot circular
    stats = np.concatenate((stats, [stats[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    
    ax10.plot(angles, stats, color='red', linewidth=2)
    ax10.fill(angles, stats, color='red', alpha=0.25)
    
    ax10.set_title('Chart 10: Champion F1-Scores (Radar)', fontsize=16, weight='bold', y=1.1)
    # Set the category labels
    ax10.set_xticks(angles[:-1])
    ax10.set_xticklabels(labels, size=12)
    ax10.set_ylim(0.75, 1.0)

    # --- Plot 11 (Right): Category F1-Scores of Top Models ---
    ax11 = fig.add_subplot(gs[2])
    sns.stripplot(data=df_plot11, y='Model Configuration', x='F1-Score', hue='Category', ax=ax11,
                  orient='h', size=10, jitter=0.15, palette='deep')
    
    ax11.set_title('Chart 11: Category F1-Scores of Top Models', fontsize=16, weight='bold')
    ax11.set_xlabel('F1-Score', fontsize=12, weight='bold')
    ax11.set_ylabel('Model Configuration', fontsize=12, weight='bold')
    ax11.set_xlim(0.68, 1.0)
    ax11.legend(title='Category', title_fontsize='13', fontsize='11', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax11.grid(axis='x', linestyle='--')

    # --- Final Touches and Save ---
    plt.tight_layout(pad=3.0)
    save_path = "plot_final_summary_charts.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Final summary plots saved successfully to '{save_path}'")
    plt.show()

# --- Main execution block ---
if __name__ == '__main__':
    # Create and save the first plot using the hardcoded data
    #create_plot_1()
    #create_plot_2_and_3()
    #create_plot_4()
    #create_plots_5_and_6()
    #create_plot_6_1()
    #create_plot_7()
    #create_plots_7_1()
    #create_final_plot_7_2()
    #create_plot_8_detailed_benchmark()
    #create_summary_plots()
    create_final_summary_plots()
