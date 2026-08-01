import numpy as np
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
BASE_PATH = '/Volumes/Reading/Projects/Empathy-LLM/'
OUTPUT_PATH = BASE_PATH + 'visualizations/'

EMOTIONS = ['afraid', 'angry', 'anxious', 'devastated', 'lonely', 'sad', 
            'terrified', 'furious', 'grateful', 'hopeful', 'faithful']

IMPORTANT_LAYERS = [2, 9, 15, 28]

def load_embeddings():
    """Load prompt and utterance embeddings"""
    print("Loading embeddings...")
    
    # Load prompt embeddings
    with open(BASE_PATH + 'prompt_embeddings.json', 'r') as f:
        prompt_embs = json.load(f)
    
    # Load utterance embeddings (from NPZ if saved, or need to load from notebook output)
    # For now, we'll load from the existing matrices and reconstruct
    print("✓ Loaded prompt embeddings")
    
    return prompt_embs

def calculate_cross_context_similarity(layer=2):
    """Calculate similarity between all prompt-utterance context pairs"""
    print(f"\nCalculating cross-context similarity for Layer {layer}...")
    
    # Load embeddings
    with open(BASE_PATH + 'prompt_embeddings.json', 'r') as f:
        prompt_embs = json.load(f)
    
    # We need utterance embeddings - let's extract from saved data
    # Load the metadata to understand structure
    with open(BASE_PATH + 'prompt_utterance_similarity_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Load existing matrices to get mean similarities
    matrices_npz = np.load(BASE_PATH + 'prompt_utterance_similarity_matrices.npz')
    
    # Create cross-context similarity matrix
    cross_sim_matrix = np.zeros((len(EMOTIONS), len(EMOTIONS)))
    
    # For each prompt context
    for i, prompt_context in enumerate(EMOTIONS):
        # Get mean embedding for all prompts in this context at this layer
        prompt_embs_list = []
        for prompt_idx_str in prompt_embs.get(prompt_context, {}):
            layer_str = str(layer)
            if layer_str in prompt_embs[prompt_context][prompt_idx_str]:
                prompt_embs_list.append(np.array(prompt_embs[prompt_context][prompt_idx_str][layer_str]))
        
        if not prompt_embs_list:
            continue
        
        prompt_mean = np.mean(prompt_embs_list, axis=0)
        
        # For each utterance context
        for j, utterance_context in enumerate(EMOTIONS):
            # Get the within-context matrix if it exists
            key = f"{utterance_context}_layer{layer}"
            
            if key in matrices_npz.files:
                # Use the mean of the matrix as approximation
                # This is the mean similarity between all prompts and utterances in that context
                if prompt_context == utterance_context:
                    # Within-context: use the actual matrix mean
                    cross_sim_matrix[i, j] = matrices_npz[key].mean()
                else:
                    # Cross-context: need to calculate
                    # For now, use a placeholder - we need utterance embeddings
                    cross_sim_matrix[i, j] = np.nan
    
    return cross_sim_matrix

def calculate_cross_context_from_stats():
    """Calculate cross-context analysis from existing statistics"""
    print("\nAnalyzing within-context vs cross-context patterns...")
    
    # Load statistics
    stats_df = pd.read_csv(BASE_PATH + 'prompt_utterance_similarity_stats.csv')
    
    results = []
    
    for layer in IMPORTANT_LAYERS:
        layer_data = stats_df[stats_df['layer'] == layer]
        
        # Get within-context similarities
        within_context_sims = []
        for context in EMOTIONS:
            context_data = layer_data[layer_data['context'] == context]
            if len(context_data) > 0:
                within_context_sims.append({
                    'context': context,
                    'layer': layer,
                    'mean_similarity': context_data['mean_similarity'].values[0],
                    'std_similarity': context_data['std_similarity'].values[0]
                })
        
        results.append({
            'layer': layer,
            'within_context_mean': np.mean([x['mean_similarity'] for x in within_context_sims]),
            'within_context_std': np.std([x['mean_similarity'] for x in within_context_sims]),
            'contexts': within_context_sims
        })
    
    return results

def create_within_context_heatmap():
    """Create heatmap showing within-context similarities"""
    print("\nCreating within-context similarity heatmap...")
    
    stats_df = pd.read_csv(BASE_PATH + 'prompt_utterance_similarity_stats.csv')
    
    # Create matrix: emotions x layers
    data_matrix = np.zeros((len(EMOTIONS), len(IMPORTANT_LAYERS)))
    
    for i, emotion in enumerate(EMOTIONS):
        for j, layer in enumerate(IMPORTANT_LAYERS):
            layer_data = stats_df[(stats_df['context'] == emotion) & (stats_df['layer'] == layer)]
            if len(layer_data) > 0:
                data_matrix[i, j] = layer_data['mean_similarity'].values[0]
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data_matrix,
                xticklabels=[f'Layer {l}' for l in IMPORTANT_LAYERS],
                yticklabels=EMOTIONS,
                cmap='RdYlGn',
                annot=True,
                fmt='.3f',
                cbar_kws={'label': 'Within-Context Similarity'},
                vmin=0.8, vmax=1.0)
    
    plt.title('Within-Context Similarity: Prompts vs Utterances (Same Emotion)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Emotion Context', fontsize=12)
    plt.tight_layout()
    
    output_file = OUTPUT_PATH + 'within_context_similarity_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_file}")
    plt.close()

def analyze_discrimination():
    """Analyze if the model can discriminate between emotions"""
    print("\n" + "="*80)
    print("EMOTION DISCRIMINATION ANALYSIS")
    print("="*80)
    
    stats_df = pd.read_csv(BASE_PATH + 'prompt_utterance_similarity_stats.csv')
    
    print("\nWITHIN-CONTEXT SIMILARITY (Same Emotion):")
    print("-" * 80)
    
    for layer in IMPORTANT_LAYERS:
        layer_data = stats_df[stats_df['layer'] == layer]
        mean_sim = layer_data['mean_similarity'].mean()
        std_sim = layer_data['mean_similarity'].std()
        min_sim = layer_data['mean_similarity'].min()
        max_sim = layer_data['mean_similarity'].max()
        
        print(f"\nLayer {layer}:")
        print(f"  Mean: {mean_sim:.4f}")
        print(f"  Std:  {std_sim:.4f}")
        print(f"  Range: [{min_sim:.4f}, {max_sim:.4f}]")
        print(f"  Contexts: {', '.join(layer_data.nsmallest(3, 'mean_similarity')['context'].values)} (lowest)")
        print(f"            {', '.join(layer_data.nlargest(3, 'mean_similarity')['context'].values)} (highest)")
    
    print("\n" + "="*80)
    print("KEY FINDING:")
    print("="*80)
    print("\n⚠️  We only calculated WITHIN-context similarity!")
    print("   (e.g., sad prompts vs sad utterances)")
    print("\n❌ We did NOT calculate CROSS-context similarity!")
    print("   (e.g., sad prompts vs happy utterances)")
    print("\n💡 To determine if the model can discriminate emotions, we need:")
    print("   - Within-context similarity (what we have)")
    print("   - Cross-context similarity (what we need)")
    print("   - Comparison: Is within > cross?")
    print("\n" + "="*80)

# Main execution
if __name__ == "__main__":
    print("="*80)
    print("CROSS-CONTEXT SIMILARITY ANALYSIS")
    print("="*80)
    
    # Analyze what we have
    analyze_discrimination()
    
    # Create visualization of within-context similarities
    create_within_context_heatmap()
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("\nTo complete the analysis, we need to:")
    print("1. Calculate cross-context similarity matrices")
    print("   (e.g., afraid prompts × grateful utterances)")
    print("2. Compare within-context vs cross-context similarities")
    print("3. Determine if model can discriminate between emotions")
    print("\nThis requires re-running the similarity calculation with")
    print("all prompt-utterance context combinations (11×11 = 121 matrices)")
    print("\n" + "="*80)
