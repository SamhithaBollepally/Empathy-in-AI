import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
BASE_PATH = '/Volumes/Reading/Projects/Empathy-LLM/'
OUTPUT_PATH = BASE_PATH + 'visualizations/'

EMOTIONS = ['afraid', 'angry', 'anxious', 'devastated', 'lonely', 'sad', 
            'terrified', 'furious', 'grateful', 'hopeful', 'faithful']

# Important layers to analyze
IMPORTANT_LAYERS = [2, 9, 15, 28]

def load_all_embeddings():
    """Load prompt and utterance embeddings"""
    print("Loading embeddings...")
    
    # Load prompt embeddings
    with open(BASE_PATH + 'prompt_embeddings.json', 'r') as f:
        prompt_embs = json.load(f)
    
    # Organize prompt embeddings by context
    prompt_data = defaultdict(lambda: defaultdict(dict))
    for context in prompt_embs:
        for prompt_idx_str in prompt_embs[context]:
            prompt_idx = int(prompt_idx_str)
            for layer_idx_str in prompt_embs[context][prompt_idx_str]:
                layer_idx = int(layer_idx_str)
                prompt_data[context][prompt_idx][layer_idx] = np.array(
                    prompt_embs[context][prompt_idx_str][layer_idx_str]
                )
    
    print(f"✓ Loaded prompt embeddings for {len(prompt_data)} contexts")
    
    # Load utterance embeddings from response_prompts.csv
    # We need to re-extract these or load from saved file
    # For now, let's check if we have them saved
    try:
        utterance_embs_npz = np.load(BASE_PATH + 'utterance_embeddings_all_layers.npz', allow_pickle=True)
        print("✓ Loaded utterance embeddings from file")
        
        # Organize utterance embeddings
        utterance_data = defaultdict(lambda: defaultdict(dict))
        for key in utterance_embs_npz.files:
            parts = key.rsplit('_', 2)
            context = parts[0]
            utt_idx = int(parts[1])
            layer_idx = int(parts[2])
            utterance_data[context][utt_idx][layer_idx] = utterance_embs_npz[key]
        
        print(f"✓ Loaded utterance embeddings for {len(utterance_data)} contexts")
        
    except FileNotFoundError:
        print("⚠ Utterance embeddings file not found!")
        print("  We need to extract utterance embeddings first.")
        print("  Please run the utterance extraction from the notebook.")
        return None, None
    
    return prompt_data, utterance_data

def calculate_cross_context_matrices(prompt_data, utterance_data, layer):
    """Calculate cross-context similarity for all emotion pairs at a given layer"""
    print(f"\n  Processing Layer {layer}...")
    
    # Create 11x11 matrix for this layer
    cross_context_matrix = np.zeros((len(EMOTIONS), len(EMOTIONS)))
    
    for i, prompt_context in enumerate(EMOTIONS):
        for j, utterance_context in enumerate(EMOTIONS):
            # Get all prompt embeddings for this context at this layer
            prompt_embs = []
            for p_idx in sorted(prompt_data[prompt_context].keys()):
                if layer in prompt_data[prompt_context][p_idx]:
                    prompt_embs.append(prompt_data[prompt_context][p_idx][layer])
            
            # Get all utterance embeddings for this context at this layer
            utterance_embs = []
            for u_idx in sorted(utterance_data[utterance_context].keys()):
                if layer in utterance_data[utterance_context][u_idx]:
                    utterance_embs.append(utterance_data[utterance_context][u_idx][layer])
            
            if not prompt_embs or not utterance_embs:
                cross_context_matrix[i, j] = np.nan
                continue
            
            # Convert to arrays
            prompt_matrix = np.array(prompt_embs)
            utterance_matrix = np.array(utterance_embs)
            
            # Calculate mean pairwise similarity
            similarity_matrix = cosine_similarity(prompt_matrix, utterance_matrix)
            cross_context_matrix[i, j] = similarity_matrix.mean()
    
    return cross_context_matrix

def analyze_discrimination(results):
    """Analyze emotion discrimination capability"""
    print("\n" + "="*80)
    print("EMOTION DISCRIMINATION ANALYSIS")
    print("="*80)
    
    for layer in IMPORTANT_LAYERS:
        if layer not in results:
            continue
        
        matrix = results[layer]
        
        # Get diagonal (within-context) and off-diagonal (cross-context)
        diagonal = np.diag(matrix)
        off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
        
        within_mean = diagonal.mean()
        cross_mean = off_diagonal.mean()
        difference = within_mean - cross_mean
        
        print(f"\nLayer {layer}:")
        print(f"  Within-context (same emotion):  {within_mean:.4f}")
        print(f"  Cross-context (diff emotions):  {cross_mean:.4f}")
        print(f"  Discrimination gap:              {difference:.4f}")
        print(f"  Relative difference:             {(difference/within_mean)*100:.2f}%")
        
        if difference < 0.01:
            print(f"  ⚠️  POOR discrimination - model cannot distinguish emotions!")
        elif difference < 0.05:
            print(f"  ⚡ WEAK discrimination - some emotion separation")
        elif difference < 0.10:
            print(f"  ✓  MODERATE discrimination")
        else:
            print(f"  ✓✓ STRONG discrimination")

def visualize_cross_context_matrices(results):
    """Create visualizations of cross-context similarity"""
    print("\nCreating visualizations...")
    
    # Create 2x2 grid for important layers
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    axes = axes.flatten()
    
    for idx, layer in enumerate(IMPORTANT_LAYERS):
        if layer not in results:
            continue
        
        ax = axes[idx]
        matrix = results[layer]
        
        # Create heatmap
        im = sns.heatmap(matrix,
                        xticklabels=EMOTIONS,
                        yticklabels=EMOTIONS,
                        cmap='RdYlGn',
                        annot=True,
                        fmt='.3f',
                        cbar_kws={'label': 'Mean Similarity'},
                        vmin=0.7, vmax=1.0,
                        ax=ax)
        
        # Calculate discrimination
        diagonal = np.diag(matrix)
        off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
        gap = diagonal.mean() - off_diagonal.mean()
        
        ax.set_title(f'Layer {layer} (Discrimination Gap: {gap:.4f})',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Utterance Context', fontsize=10)
        ax.set_ylabel('Prompt Context', fontsize=10)
        
        # Rotate labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.suptitle('Cross-Context Similarity: Prompt Context × Utterance Context',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = OUTPUT_PATH + 'cross_context_similarity_matrices.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_file}")
    plt.close()

def create_discrimination_plot(results):
    """Plot discrimination gap across layers"""
    print("\nCreating discrimination gap plot...")
    
    layers = []
    within_sims = []
    cross_sims = []
    gaps = []
    
    for layer in sorted(results.keys()):
        matrix = results[layer]
        diagonal = np.diag(matrix)
        off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
        
        layers.append(layer)
        within_sims.append(diagonal.mean())
        cross_sims.append(off_diagonal.mean())
        gaps.append(diagonal.mean() - off_diagonal.mean())
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Within vs Cross similarity
    ax1.plot(layers, within_sims, 'o-', label='Within-context (same emotion)', 
             linewidth=2, markersize=8, color='green')
    ax1.plot(layers, cross_sims, 's-', label='Cross-context (different emotions)',
             linewidth=2, markersize=8, color='orange')
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Mean Similarity', fontsize=12)
    ax1.set_title('Within-Context vs Cross-Context Similarity Across Layers',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.7, 1.0)
    
    # Plot 2: Discrimination gap
    ax2.plot(layers, gaps, 'D-', linewidth=2, markersize=8, color='blue')
    ax2.fill_between(layers, 0, gaps, alpha=0.3, color='blue')
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Discrimination Gap', fontsize=12)
    ax2.set_title('Emotion Discrimination Gap (Within - Cross)',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Add threshold lines
    ax2.axhline(y=0.01, color='orange', linestyle=':', alpha=0.5, label='Weak (0.01)')
    ax2.axhline(y=0.05, color='yellow', linestyle=':', alpha=0.5, label='Moderate (0.05)')
    ax2.axhline(y=0.10, color='green', linestyle=':', alpha=0.5, label='Strong (0.10)')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    output_file = OUTPUT_PATH + 'discrimination_gap_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_file}")
    plt.close()

def save_results(results):
    """Save cross-context similarity results"""
    print("\nSaving results...")
    
    # Save matrices as NPZ
    matrices_dict = {f"layer{layer}": results[layer] for layer in results}
    np.savez_compressed(BASE_PATH + 'cross_context_similarity_matrices.npz', **matrices_dict)
    print(f"✓ Saved cross_context_similarity_matrices.npz")
    
    # Save as CSV for easy viewing
    all_data = []
    for layer in results:
        matrix = results[layer]
        for i, prompt_ctx in enumerate(EMOTIONS):
            for j, utt_ctx in enumerate(EMOTIONS):
                all_data.append({
                    'layer': layer,
                    'prompt_context': prompt_ctx,
                    'utterance_context': utt_ctx,
                    'mean_similarity': matrix[i, j],
                    'is_within_context': (prompt_ctx == utt_ctx)
                })
    
    df = pd.DataFrame(all_data)
    df.to_csv(BASE_PATH + 'cross_context_similarity_stats.csv', index=False)
    print(f"✓ Saved cross_context_similarity_stats.csv")

# Main execution
if __name__ == "__main__":
    print("="*80)
    print("CROSS-CONTEXT SIMILARITY CALCULATION")
    print("Full 11×11 Emotion Matrix Analysis")
    print("="*80)
    
    # Load embeddings
    prompt_data, utterance_data = load_all_embeddings()
    
    if prompt_data is None or utterance_data is None:
        print("\n❌ Cannot proceed without utterance embeddings!")
        print("\nPlease ensure utterance embeddings are saved from the notebook.")
        exit(1)
    
    # Calculate cross-context similarity for important layers
    print("\nCalculating cross-context similarity matrices...")
    results = {}
    
    for layer in IMPORTANT_LAYERS:
        matrix = calculate_cross_context_matrices(prompt_data, utterance_data, layer)
        results[layer] = matrix
        print(f"    ✓ Layer {layer} complete")
    
    # Analyze discrimination
    analyze_discrimination(results)
    
    # Create visualizations
    visualize_cross_context_matrices(results)
    create_discrimination_plot(results)
    
    # Save results
    save_results(results)
    
    print("\n" + "="*80)
    print("✓ Cross-context analysis complete!")
    print("="*80)
