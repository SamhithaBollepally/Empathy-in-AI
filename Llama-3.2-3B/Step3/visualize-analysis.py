import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# Configuration
BASE_PATH = '/Volumes/Reading/Projects/Empathy-LLM/'
OUTPUT_PATH = BASE_PATH + 'visualizations/'

EMOTIONS = ['afraid', 'angry', 'anxious', 'devastated', 'lonely', 'sad', 
            'terrified', 'furious', 'grateful', 'hopeful', 'faithful']

# Create output directory
import os
os.makedirs(OUTPUT_PATH, exist_ok=True)

def load_data():
    """Load all necessary data"""
    print("Loading data...")
    
    # Load probing results
    with open(BASE_PATH + 'probing_results.json', 'r') as f:
        probing_data = json.load(f)
    
    # Load similarity analysis
    similarity_df = pd.read_csv(BASE_PATH + 'layer_accuracy_summary.csv')
    
    print(f"✓ Loaded probing results")
    print(f"✓ Loaded similarity results")
    
    return probing_data, similarity_df

def create_probing_heatmap(probing_data):
    """Create heatmap of probing accuracy per emotion per layer"""
    print("\n1. Creating probing accuracy heatmap...")
    
    # Prepare data matrix
    data_matrix = []
    for emotion in EMOTIONS:
        if emotion in probing_data['context_layer_accuracies']:
            layer_accs = probing_data['context_layer_accuracies'][emotion]
            row = [layer_accs[str(i)] for i in range(29)]
            data_matrix.append(row)
    
    data_matrix = np.array(data_matrix)
    
    # Create figure
    plt.figure(figsize=(16, 8))
    sns.heatmap(data_matrix, 
                xticklabels=range(29),
                yticklabels=EMOTIONS,
                cmap='RdYlGn',
                annot=False,
                fmt='.2f',
                cbar_kws={'label': 'Probing Accuracy'},
                vmin=0, vmax=1)
    
    plt.title('Probing Classifier Accuracy per Emotion per Layer', fontsize=16, fontweight='bold')
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Emotion', fontsize=12)
    plt.tight_layout()
    
    output_file = OUTPUT_PATH + 'probing_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {output_file}")
    plt.close()

def create_similarity_heatmap(similarity_df):
    """Create heatmap of response similarity accuracy per emotion per layer"""
    print("\n2. Creating similarity accuracy heatmap...")
    
    # Prepare data matrix
    data_matrix = []
    for emotion in EMOTIONS:
        col = f'acc_{emotion}'
        row = similarity_df[col].values
        data_matrix.append(row)
    
    data_matrix = np.array(data_matrix)
    
    # Create figure
    plt.figure(figsize=(16, 8))
    sns.heatmap(data_matrix,
                xticklabels=range(29),
                yticklabels=EMOTIONS,
                cmap='RdYlGn',
                annot=False,
                fmt='.2f',
                cbar_kws={'label': 'Similarity Accuracy'},
                vmin=0, vmax=1)
    
    plt.title('Response Similarity Accuracy per Emotion per Layer', fontsize=16, fontweight='bold')
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Emotion', fontsize=12)
    plt.tight_layout()
    
    output_file = OUTPUT_PATH + 'similarity_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {output_file}")
    plt.close()

def create_combined_heatmap(probing_data, similarity_df):
    """Create side-by-side comparison heatmap"""
    print("\n3. Creating combined comparison heatmap...")
    
    # Prepare probing data
    probing_matrix = []
    for emotion in EMOTIONS:
        if emotion in probing_data['context_layer_accuracies']:
            layer_accs = probing_data['context_layer_accuracies'][emotion]
            row = [layer_accs[str(i)] for i in range(29)]
            probing_matrix.append(row)
    probing_matrix = np.array(probing_matrix)
    
    # Prepare similarity data
    similarity_matrix = []
    for emotion in EMOTIONS:
        col = f'acc_{emotion}'
        row = similarity_df[col].values
        similarity_matrix.append(row)
    similarity_matrix = np.array(similarity_matrix)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Probing heatmap
    sns.heatmap(probing_matrix,
                xticklabels=range(29),
                yticklabels=EMOTIONS,
                cmap='RdYlGn',
                annot=False,
                cbar_kws={'label': 'Accuracy'},
                vmin=0, vmax=1,
                ax=axes[0])
    axes[0].set_title('Probing Classifier Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Layer', fontsize=11)
    axes[0].set_ylabel('Emotion', fontsize=11)
    
    # Similarity heatmap
    sns.heatmap(similarity_matrix,
                xticklabels=range(29),
                yticklabels=EMOTIONS,
                cmap='RdYlGn',
                annot=False,
                cbar_kws={'label': 'Accuracy'},
                vmin=0, vmax=1,
                ax=axes[1])
    axes[1].set_title('Response Similarity Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Layer', fontsize=11)
    axes[1].set_ylabel('Emotion', fontsize=11)
    
    plt.suptitle('Probing vs Response Similarity: Layer-wise Accuracy Comparison', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = OUTPUT_PATH + 'combined_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {output_file}")
    plt.close()

def create_violin_plots(probing_data, similarity_df):
    """Create violin plots showing distribution across layers"""
    print("\n4. Creating violin plots...")
    
    # Prepare data for violin plots
    probing_distributions = []
    similarity_distributions = []
    
    for emotion in EMOTIONS:
        # Probing data
        if emotion in probing_data['context_layer_accuracies']:
            layer_accs = probing_data['context_layer_accuracies'][emotion]
            values = [layer_accs[str(i)] for i in range(29)]
            for val in values:
                probing_distributions.append({
                    'emotion': emotion,
                    'accuracy': val,
                    'type': 'Probing'
                })
        
        # Similarity data
        col = f'acc_{emotion}'
        values = similarity_df[col].values
        for val in values:
            similarity_distributions.append({
                'emotion': emotion,
                'accuracy': val,
                'type': 'Similarity'
            })
    
    # Combine data
    all_data = pd.DataFrame(probing_distributions + similarity_distributions)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Probing violin plot
    probing_df = all_data[all_data['type'] == 'Probing']
    sns.violinplot(data=probing_df, x='emotion', y='accuracy', 
                   palette='Set2', ax=axes[0])
    axes[0].set_title('Probing Accuracy Distribution Across Layers', 
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Emotion', fontsize=11)
    axes[0].set_ylabel('Accuracy', fontsize=11)
    axes[0].set_ylim(0, 1)
    axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% baseline')
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)
    
    # Similarity violin plot
    similarity_df_plot = all_data[all_data['type'] == 'Similarity']
    sns.violinplot(data=similarity_df_plot, x='emotion', y='accuracy',
                   palette='Set3', ax=axes[1])
    axes[1].set_title('Response Similarity Accuracy Distribution Across Layers',
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Emotion', fontsize=11)
    axes[1].set_ylabel('Accuracy', fontsize=11)
    axes[1].set_ylim(0, 1)
    axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% baseline')
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    output_file = OUTPUT_PATH + 'violin_plots.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {output_file}")
    plt.close()

def create_layer_trends(probing_data, similarity_df):
    """Create line plots showing trends across layers"""
    print("\n5. Creating layer trend plots...")
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Probing trends
    for emotion in EMOTIONS:
        if emotion in probing_data['context_layer_accuracies']:
            layer_accs = probing_data['context_layer_accuracies'][emotion]
            values = [layer_accs[str(i)] for i in range(29)]
            axes[0].plot(range(29), values, marker='o', label=emotion, alpha=0.7)
    
    axes[0].set_title('Probing Accuracy Trends Across Layers', 
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Layer', fontsize=11)
    axes[0].set_ylabel('Accuracy', fontsize=11)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    
    # Similarity trends
    for emotion in EMOTIONS:
        col = f'acc_{emotion}'
        values = similarity_df[col].values
        axes[1].plot(range(29), values, marker='s', label=emotion, alpha=0.7)
    
    axes[1].set_title('Response Similarity Accuracy Trends Across Layers',
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Layer', fontsize=11)
    axes[1].set_ylabel('Accuracy', fontsize=11)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    output_file = OUTPUT_PATH + 'layer_trends.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {output_file}")
    plt.close()

def create_correlation_scatter(probing_data, similarity_df):
    """Create scatter plots showing correlation for each emotion"""
    print("\n6. Creating correlation scatter plots...")
    
    # Create 3x4 grid for 11 emotions
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for idx, emotion in enumerate(EMOTIONS):
        ax = axes[idx]
        
        # Prepare data
        probing_vals = []
        similarity_vals = []
        
        if emotion in probing_data['context_layer_accuracies']:
            layer_accs = probing_data['context_layer_accuracies'][emotion]
            col = f'acc_{emotion}'
            
            for layer in range(29):
                probing_vals.append(layer_accs[str(layer)])
                similarity_vals.append(
                    similarity_df[similarity_df['layer'] == layer][col].values[0]
                )
            
            # Calculate correlation
            rho, p_value = spearmanr(probing_vals, similarity_vals)
            
            # Scatter plot
            ax.scatter(probing_vals, similarity_vals, alpha=0.6, s=100)
            
            # Add trend line
            z = np.polyfit(probing_vals, similarity_vals, 1)
            p = np.poly1d(z)
            ax.plot(probing_vals, p(probing_vals), "r--", alpha=0.5)
            
            # Labels
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            ax.set_title(f'{emotion}\nρ={rho:.3f} {sig}', fontweight='bold')
            ax.set_xlabel('Probing Accuracy', fontsize=9)
            ax.set_ylabel('Similarity Accuracy', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
    
    # Hide extra subplot
    axes[-1].axis('off')
    
    plt.suptitle('Correlation: Probing vs Similarity Accuracy (per emotion)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = OUTPUT_PATH + 'correlation_scatter.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {output_file}")
    plt.close()

def create_best_layer_comparison(probing_data, similarity_df):
    """Compare best layers from probing vs similarity"""
    print("\n7. Creating best layer comparison...")
    
    # Find best layers
    best_layers_probing = {}
    best_layers_similarity = {}
    
    for emotion in EMOTIONS:
        # Probing best layer
        if emotion in probing_data['context_layer_accuracies']:
            layer_accs = probing_data['context_layer_accuracies'][emotion]
            best_layer = max(layer_accs.items(), key=lambda x: x[1])
            best_layers_probing[emotion] = int(best_layer[0])
        
        # Similarity best layer
        col = f'acc_{emotion}'
        best_row = similarity_df.loc[similarity_df[col].idxmax()]
        best_layers_similarity[emotion] = int(best_row['layer'])
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(EMOTIONS))
    width = 0.35
    
    probing_layers = [best_layers_probing.get(e, 0) for e in EMOTIONS]
    similarity_layers = [best_layers_similarity.get(e, 0) for e in EMOTIONS]
    
    bars1 = ax.bar(x - width/2, probing_layers, width, label='Probing', alpha=0.8)
    bars2 = ax.bar(x + width/2, similarity_layers, width, label='Similarity', alpha=0.8)
    
    ax.set_xlabel('Emotion', fontsize=12)
    ax.set_ylabel('Best Layer', fontsize=12)
    ax.set_title('Best Layer Comparison: Probing vs Similarity', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(EMOTIONS, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 28)
    
    plt.tight_layout()
    
    output_file = OUTPUT_PATH + 'best_layer_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {output_file}")
    plt.close()

# Main execution
if __name__ == "__main__":
    print("="*80)
    print("VISUALIZATION: PROBING vs SIMILARITY ANALYSIS")
    print("="*80)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Load data
    probing_data, similarity_df = load_data()
    
    # Create visualizations
    create_probing_heatmap(probing_data)
    create_similarity_heatmap(similarity_df)
    create_combined_heatmap(probing_data, similarity_df)
    create_violin_plots(probing_data, similarity_df)
    create_layer_trends(probing_data, similarity_df)
    create_correlation_scatter(probing_data, similarity_df)
    create_best_layer_comparison(probing_data, similarity_df)
    
    print("\n" + "="*80)
    print(f"✓ All visualizations saved to {OUTPUT_PATH}")
    print("="*80)
    print("\nGenerated files:")
    print("  1. probing_heatmap.png - Probing accuracy heatmap")
    print("  2. similarity_heatmap.png - Similarity accuracy heatmap")
    print("  3. combined_heatmap.png - Side-by-side comparison")
    print("  4. violin_plots.png - Distribution across layers")
    print("  5. layer_trends.png - Trends across layers")
    print("  6. correlation_scatter.png - Correlation per emotion")
    print("  7. best_layer_comparison.png - Best layer comparison")
    print("\n✓ Visualization complete!")
