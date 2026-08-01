import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
BASE_PATH = '/Volumes/Reading/Projects/Empathy-LLM/'
OUTPUT_PATH = BASE_PATH + 'visualizations/'

EMOTIONS = ['afraid', 'angry', 'anxious', 'devastated', 'lonely', 'sad', 
            'terrified', 'furious', 'grateful', 'hopeful', 'faithful']

# Important layers to visualize
IMPORTANT_LAYERS = [2, 9, 15, 28]
LAYER_DESCRIPTIONS = {
    2: 'Layer 2 (Best - 99.87% similarity)',
    9: 'Layer 9 (Mid - 98.57% similarity)',
    15: 'Layer 15 (Deep - 96.88% similarity)',
    28: 'Layer 28 (Final - 80% similarity)'
}

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)

def load_matrices():
    """Load similarity matrices from NPZ file"""
    print("Loading similarity matrices...")
    matrices_npz = np.load(BASE_PATH + 'prompt_utterance_similarity_matrices.npz')
    
    matrices = {}
    for key in matrices_npz.files:
        matrices[key] = matrices_npz[key]
    
    print(f"✓ Loaded {len(matrices)} matrices")
    return matrices

def plot_single_matrix(matrix, context, layer, save_path):
    """Plot a single similarity matrix"""
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(matrix, 
                cmap='RdYlGn',
                vmin=0, vmax=1,
                cbar_kws={'label': 'Cosine Similarity'},
                xticklabels=False,
                yticklabels=False)
    
    plt.title(f'{context.upper()} - {LAYER_DESCRIPTIONS[layer]}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Utterance Index', fontsize=11)
    plt.ylabel('Prompt Index', fontsize=11)
    
    # Add statistics
    mean_sim = matrix.mean()
    std_sim = matrix.std()
    plt.text(0.02, 0.98, f'Mean: {mean_sim:.4f}\nStd: {std_sim:.4f}',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             verticalalignment='top',
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_layer_comparison_grid(matrices, context):
    """Plot all important layers for one context in a grid"""
    print(f"\n  Creating grid for {context}...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for idx, layer in enumerate(IMPORTANT_LAYERS):
        key = f"{context}_layer{layer}"
        if key not in matrices:
            continue
        
        matrix = matrices[key]
        ax = axes[idx]
        
        # Create heatmap
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        
        ax.set_title(LAYER_DESCRIPTIONS[layer], fontsize=12, fontweight='bold')
        ax.set_xlabel('Utterance Index', fontsize=10)
        ax.set_ylabel('Prompt Index', fontsize=10)
        
        # Add statistics
        mean_sim = matrix.mean()
        std_sim = matrix.std()
        ax.text(0.02, 0.98, f'Mean: {mean_sim:.4f}\nStd: {std_sim:.4f}',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top',
                fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Similarity', fontsize=9)
    
    plt.suptitle(f'{context.upper()}: Pairwise Similarity Across Layers',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = OUTPUT_PATH + f'pairwise_matrix_{context}_grid.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved {output_file}")
    plt.close()

def plot_context_comparison(matrices, layer):
    """Plot all contexts for one layer in a grid"""
    print(f"\n  Creating context comparison for Layer {layer}...")
    
    # Create 3x4 grid for 11 emotions
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for idx, context in enumerate(EMOTIONS):
        key = f"{context}_layer{layer}"
        if key not in matrices:
            continue
        
        matrix = matrices[key]
        ax = axes[idx]
        
        # Create heatmap
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        
        mean_sim = matrix.mean()
        ax.set_title(f'{context}\n(Mean: {mean_sim:.4f})', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Utterance', fontsize=9)
        ax.set_ylabel('Prompt', fontsize=9)
        
        # Reduce tick labels
        ax.set_xticks([0, 50, 99])
        ax.set_yticks([0, 50, 99])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Sim', fontsize=8)
    
    # Hide extra subplot
    axes[-1].axis('off')
    
    plt.suptitle(f'{LAYER_DESCRIPTIONS[layer]}: All Contexts',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = OUTPUT_PATH + f'pairwise_matrix_layer{layer}_all_contexts.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved {output_file}")
    plt.close()

def plot_diagonal_analysis(matrices):
    """Analyze diagonal values (same prompt-utterance pairs)"""
    print("\n  Creating diagonal analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, layer in enumerate(IMPORTANT_LAYERS):
        ax = axes[idx]
        
        # Collect diagonal values for all contexts
        diagonal_data = []
        
        for context in EMOTIONS:
            key = f"{context}_layer{layer}"
            if key not in matrices:
                continue
            
            matrix = matrices[key]
            # Get diagonal (assumes prompts and utterances are aligned by index)
            diagonal = np.diag(matrix)
            
            for val in diagonal:
                diagonal_data.append({'context': context, 'similarity': val})
        
        # Create violin plot
        import pandas as pd
        df = pd.DataFrame(diagonal_data)
        
        # Plot
        parts = ax.violinplot([df[df['context'] == ctx]['similarity'].values 
                               for ctx in EMOTIONS],
                              positions=range(len(EMOTIONS)),
                              showmeans=True,
                              showmedians=True)
        
        ax.set_xticks(range(len(EMOTIONS)))
        ax.set_xticklabels(EMOTIONS, rotation=45, ha='right')
        ax.set_ylabel('Diagonal Similarity', fontsize=11)
        ax.set_title(LAYER_DESCRIPTIONS[layer], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
        
        # Add mean line
        overall_mean = df['similarity'].mean()
        ax.axhline(y=overall_mean, color='red', linestyle='--', 
                  alpha=0.5, label=f'Mean: {overall_mean:.4f}')
        ax.legend()
    
    plt.suptitle('Diagonal Similarity Analysis (Same Prompt-Utterance Pairs)',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = OUTPUT_PATH + 'pairwise_matrix_diagonal_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved {output_file}")
    plt.close()

def create_summary_heatmap(matrices):
    """Create summary heatmap showing mean similarity for each context-layer combination"""
    print("\n  Creating summary heatmap...")
    
    # Prepare data matrix
    data_matrix = np.zeros((len(EMOTIONS), len(IMPORTANT_LAYERS)))
    
    for i, context in enumerate(EMOTIONS):
        for j, layer in enumerate(IMPORTANT_LAYERS):
            key = f"{context}_layer{layer}"
            if key in matrices:
                data_matrix[i, j] = matrices[key].mean()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data_matrix,
                xticklabels=[f'L{l}' for l in IMPORTANT_LAYERS],
                yticklabels=EMOTIONS,
                cmap='RdYlGn',
                annot=True,
                fmt='.4f',
                cbar_kws={'label': 'Mean Similarity'},
                vmin=0.7, vmax=1.0)
    
    plt.title('Mean Pairwise Similarity: Context × Layer',
              fontsize=14, fontweight='bold')
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Context', fontsize=12)
    plt.tight_layout()
    
    output_file = OUTPUT_PATH + 'pairwise_matrix_summary_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved {output_file}")
    plt.close()

# Main execution
if __name__ == "__main__":
    print("="*80)
    print("PAIRWISE SIMILARITY MATRIX VISUALIZATION")
    print("="*80)
    
    # Load matrices
    matrices = load_matrices()
    
    print("\n1. Creating layer comparison grids for each context...")
    for context in EMOTIONS:
        plot_layer_comparison_grid(matrices, context)
    
    print("\n2. Creating context comparison for each layer...")
    for layer in IMPORTANT_LAYERS:
        plot_context_comparison(matrices, layer)
    
    print("\n3. Creating diagonal analysis...")
    plot_diagonal_analysis(matrices)
    
    print("\n4. Creating summary heatmap...")
    create_summary_heatmap(matrices)
    
    print("\n" + "="*80)
    print(f"✓ All visualizations saved to {OUTPUT_PATH}")
    print("="*80)
    print("\nGenerated files:")
    print("  - pairwise_matrix_{context}_grid.png (11 files)")
    print("  - pairwise_matrix_layer{layer}_all_contexts.png (4 files)")
    print("  - pairwise_matrix_diagonal_analysis.png")
    print("  - pairwise_matrix_summary_heatmap.png")
    print("\n✓ Visualization complete!")
