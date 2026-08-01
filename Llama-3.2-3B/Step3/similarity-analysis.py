import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

# Configuration
BASE_PATH = '/Volumes/Reading/Projects/Empathy-LLM/'

BEST_LAYERS = {
    'afraid': 6, 'angry': 5, 'anxious': 9, 'devastated': 8,
    'lonely': 4, 'sad': 7, 'terrified': 1, 'furious': 7,
    'grateful': 15, 'hopeful': 12, 'faithful': 5
}

EMOTIONS = list(BEST_LAYERS.keys())

def load_embeddings():
    """Load response and context embeddings"""
    print("Loading embeddings...")
    
    # Load response embeddings (all layers)
    response_embs_npz = np.load(BASE_PATH + 'response_embeddings.npz', allow_pickle=True)
    
    # Load context embeddings
    with open(BASE_PATH + 'context_embeddings.json', 'r') as f:
        context_data = json.load(f)
    
    # Convert context embeddings to arrays
    context_embs = {}
    for context, layers_dict in context_data.items():
        for layer_str, embedding in layers_dict.items():
            layer_idx = int(layer_str)
            context_embs[f"{context}_{layer_idx}"] = np.array(embedding)
    
    print(f"✓ Loaded {len(response_embs_npz.files)} response embeddings")
    print(f"✓ Loaded {len(context_embs)} context prototypes")
    
    return response_embs_npz, context_embs

def analyze_all_layers(response_embs_npz, context_embs):
    """Analyze cosine similarity across all layers"""
    print("\nAnalyzing similarity across all 29 layers...")
    
    # Organize response embeddings by context, response_idx, and layer
    response_data = {}
    for key in response_embs_npz.files:
        parts = key.rsplit('_', 2)
        context = parts[0]
        response_idx = int(parts[1])
        layer_idx = int(parts[2])
        
        if context not in response_data:
            response_data[context] = {}
        if response_idx not in response_data[context]:
            response_data[context][response_idx] = {}
        
        response_data[context][response_idx][layer_idx] = response_embs_npz[key]
    
    # Results storage
    all_results = []
    layer_accuracies = {layer: [] for layer in range(29)}
    
    # Analyze each layer
    for layer in range(29):
        print(f"  Processing layer {layer}...")
        
        layer_results = []
        
        for context in EMOTIONS:
            if context not in response_data:
                continue
            
            for response_idx, layers_dict in response_data[context].items():
                if layer not in layers_dict:
                    continue
                
                response_emb = layers_dict[layer].reshape(1, -1)
                
                # Compare against all emotion prototypes at this layer
                sims = {}
                for emotion in EMOTIONS:
                    proto_key = f"{emotion}_{layer}"
                    if proto_key in context_embs:
                        proto = context_embs[proto_key].reshape(1, -1)
                        sims[emotion] = cosine_similarity(response_emb, proto)[0][0]
                
                if not sims:
                    continue
                
                predicted_emotion = max(sims, key=sims.get)
                correct = (predicted_emotion == context)
                
                result = {
                    'layer': layer,
                    'context': context,
                    'response_idx': response_idx,
                    'predicted_emotion': predicted_emotion,
                    'correct': correct,
                    'correct_sim': sims[context],
                    **{f"sim_{e}": sims[e] for e in EMOTIONS}
                }
                
                layer_results.append(result)
                all_results.append(result)
        
        # Calculate layer accuracy
        if layer_results:
            layer_acc = sum(r['correct'] for r in layer_results) / len(layer_results)
            layer_accuracies[layer] = layer_acc
            print(f"    Layer {layer} accuracy: {layer_acc:.2%}")
    
    return all_results, layer_accuracies

def save_results(all_results, layer_accuracies):
    """Save analysis results"""
    print("\nSaving results...")
    
    # Save detailed results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(BASE_PATH + 'similarity_all_layers.csv', index=False)
    print(f"✓ Saved detailed results to similarity_all_layers.csv")
    
    # Save layer-wise summary
    layer_summary = []
    for layer in range(29):
        if layer_accuracies[layer]:
            layer_df = results_df[results_df['layer'] == layer]
            
            # Per-context accuracy for this layer
            context_accs = layer_df.groupby('context')['correct'].mean().to_dict()
            
            summary = {
                'layer': layer,
                'overall_accuracy': layer_accuracies[layer],
                'total_samples': len(layer_df),
                **{f"acc_{ctx}": context_accs.get(ctx, 0) for ctx in EMOTIONS}
            }
            layer_summary.append(summary)
    
    summary_df = pd.DataFrame(layer_summary)
    summary_df.to_csv(BASE_PATH + 'layer_accuracy_summary.csv', index=False)
    print(f"✓ Saved layer summary to layer_accuracy_summary.csv")
    
    # Find best layer overall and per context
    best_layer_overall = max(layer_accuracies.items(), key=lambda x: x[1] if x[1] else 0)
    print(f"\n✓ Best layer overall: Layer {best_layer_overall[0]} ({best_layer_overall[1]:.2%})")
    
    # Best layer per context
    print("\n✓ Best layer per context:")
    for context in EMOTIONS:
        context_df = results_df[results_df['context'] == context]
        context_layer_acc = context_df.groupby('layer')['correct'].mean()
        if len(context_layer_acc) > 0:
            best_layer = context_layer_acc.idxmax()
            best_acc = context_layer_acc.max()
            print(f"   {context:<12}: Layer {best_layer:2d} ({best_acc:.2%})")
    
    return results_df, summary_df

def plot_layer_trends(summary_df):
    """Print layer accuracy trends"""
    print("\n" + "="*70)
    print("LAYER ACCURACY TRENDS")
    print("="*70)
    
    print("\nOverall Accuracy by Layer:")
    for _, row in summary_df.iterrows():
        layer = int(row['layer'])
        acc = row['overall_accuracy']
        bar_length = int(acc * 50)
        bar = '█' * bar_length
        print(f"  Layer {layer:2d}: {acc:.2%} {bar}")
    
    print("\n" + "="*70)

# Main execution
if __name__ == "__main__":
    print("="*70)
    print("COSINE SIMILARITY ANALYSIS - ALL LAYERS")
    print("="*70)
    
    # Load embeddings
    response_embs_npz, context_embs = load_embeddings()
    
    # Analyze all layers
    all_results, layer_accuracies = analyze_all_layers(response_embs_npz, context_embs)
    
    # Save results
    results_df, summary_df = save_results(all_results, layer_accuracies)
    
    # Plot trends
    plot_layer_trends(summary_df)
    
    print("\n✓ Analysis complete!")
