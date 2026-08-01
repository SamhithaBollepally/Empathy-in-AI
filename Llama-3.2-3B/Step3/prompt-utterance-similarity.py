#%%
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import json
import os
#%%
# Configuration
# BASE_PATH = '/Volumes/Reading/Projects/Empathy-LLM/'
BASE_PATH = '/content/drive/MyDrive/Project-4/'
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_NAME = "meta-llama/Llama-3.2-3B"

EMOTIONS = ['afraid', 'angry', 'anxious', 'devastated', 'lonely', 'sad', 
            'terrified', 'furious', 'grateful', 'hopeful', 'faithful']
#%%
def setup_model():
    """Load and configure model on GPU"""
    print("Setting up model...")
    login(token=HF_TOKEN)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, low_cpu_mem_usage=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("✓ Using CPU")
    
    return model, tokenizer, device
#%%
def extract_embeddings(text, model, tokenizer, device):
    """Extract embeddings from all layers"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    
    embeddings = {}
    for layer_idx, hidden_state in enumerate(outputs.hidden_states):
        embeddings[layer_idx] = hidden_state.mean(dim=1).squeeze().cpu().numpy()
    
    return embeddings

#%%
def load_prompt_embeddings():
    """Load existing prompt embeddings from Step 1"""
    print("\nLoading prompt embeddings from Step 1...")
    
    # Load prompt data
    prompts_df = pd.read_csv(BASE_PATH + 'combined_filtered.csv')
    
    # Load embeddings
    with open(BASE_PATH + 'prompt_embeddings.json', 'r') as f:
        prompt_embs = json.load(f)
    
    # Organize by context
    # JSON structure: {context: {prompt_idx: {layer_idx: embedding}}}
    prompt_data = defaultdict(lambda: defaultdict(dict))
    
    for context in prompt_embs:
        for prompt_idx_str in prompt_embs[context]:
            prompt_idx = int(prompt_idx_str)
            for layer_idx_str in prompt_embs[context][prompt_idx_str]:
                layer_idx = int(layer_idx_str)
                prompt_data[context][prompt_idx][layer_idx] = np.array(prompt_embs[context][prompt_idx_str][layer_idx_str])
    
    print(f"✓ Loaded embeddings for {len(prompt_data)} contexts")
    for context in EMOTIONS:
        if context in prompt_data:
            print(f"   {context:<12}: {len(prompt_data[context])} prompts")
    
    return prompt_data, prompts_df
#%%
def extract_utterance_embeddings(model, tokenizer, device):
    """Extract embeddings from first utterances in response_prompts.csv"""
    print("\nExtracting utterance embeddings...")
    
    # Load response prompts
    response_df = pd.read_csv(BASE_PATH + 'response_prompts.csv')
    
    utterance_data = defaultdict(lambda: defaultdict(dict))
    
    for idx, row in response_df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(response_df)} utterances")
        
        context = row['context']
        utterance = row['first_utterance']
        
        # Extract embeddings from all layers
        embeddings = extract_embeddings(utterance, model, tokenizer, device)
        
        # Store by context and index
        for layer_idx, emb in embeddings.items():
            utterance_data[context][idx][layer_idx] = emb
    
    print(f"✓ Extracted embeddings for {len(response_df)} utterances")
    for context in EMOTIONS:
        if context in utterance_data:
            print(f"   {context:<12}: {len(utterance_data[context])} utterances")
    
    return utterance_data, response_df
#%%
def calculate_similarity_matrices(prompt_data, utterance_data):
    """Calculate pairwise similarity matrices for all layers and contexts"""
    print("\nCalculating pairwise similarity matrices...")
    
    results = {}
    
    for context in EMOTIONS:
        if context not in prompt_data or context not in utterance_data:
            print(f"  ⚠ Skipping {context} - missing data")
            continue
        
        print(f"\n  Processing {context}...")
        
        # Get prompt and utterance indices for this context
        prompt_indices = sorted(prompt_data[context].keys())
        utterance_indices = sorted(utterance_data[context].keys())
        
        n_prompts = len(prompt_indices)
        n_utterances = len(utterance_indices)
        
        print(f"    Prompts: {n_prompts}, Utterances: {n_utterances}")
        
        # Calculate similarity for each layer
        context_results = {}
        
        for layer in range(29):
            # Get all prompt embeddings for this layer
            prompt_embs = []
            for p_idx in prompt_indices:
                if layer in prompt_data[context][p_idx]:
                    prompt_embs.append(prompt_data[context][p_idx][layer])
            
            # Get all utterance embeddings for this layer
            utterance_embs = []
            for u_idx in utterance_indices:
                if layer in utterance_data[context][u_idx]:
                    utterance_embs.append(utterance_data[context][u_idx][layer])
            
            if not prompt_embs or not utterance_embs:
                continue
            
            # Convert to arrays
            prompt_matrix = np.array(prompt_embs)  # shape: (n_prompts, embedding_dim)
            utterance_matrix = np.array(utterance_embs)  # shape: (n_utterances, embedding_dim)
            
            # Calculate pairwise cosine similarity
            # Result shape: (n_prompts, n_utterances)
            similarity_matrix = cosine_similarity(prompt_matrix, utterance_matrix)
            
            context_results[layer] = {
                'matrix': similarity_matrix,
                'prompt_indices': prompt_indices,
                'utterance_indices': utterance_indices,
                'mean_similarity': similarity_matrix.mean(),
                'std_similarity': similarity_matrix.std(),
                'max_similarity': similarity_matrix.max(),
                'min_similarity': similarity_matrix.min()
            }
        
        results[context] = context_results
        print(f"    ✓ Calculated {len(context_results)} layer matrices")
    
    return results
#%%
def save_results(results):
    """Save similarity matrices and statistics"""
    print("\nSaving results...")
    
    # Save matrices as NPZ (more efficient for large matrices)
    matrices_dict = {}
    for context in results:
        for layer in results[context]:
            key = f"{context}_layer{layer}"
            matrices_dict[key] = results[context][layer]['matrix']
    
    np.savez_compressed(BASE_PATH + 'prompt_utterance_similarity_matrices.npz', **matrices_dict)
    print(f"✓ Saved similarity matrices to prompt_utterance_similarity_matrices.npz")
    
    # Save statistics as CSV
    stats_data = []
    for context in results:
        for layer in results[context]:
            stats_data.append({
                'context': context,
                'layer': layer,
                'n_prompts': len(results[context][layer]['prompt_indices']),
                'n_utterances': len(results[context][layer]['utterance_indices']),
                'mean_similarity': results[context][layer]['mean_similarity'],
                'std_similarity': results[context][layer]['std_similarity'],
                'max_similarity': results[context][layer]['max_similarity'],
                'min_similarity': results[context][layer]['min_similarity']
            })
    
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(BASE_PATH + 'prompt_utterance_similarity_stats.csv', index=False)
    print(f"✓ Saved statistics to prompt_utterance_similarity_stats.csv")
    
    # Save metadata (indices mapping)
    metadata = {}
    for context in results:
        for layer in results[context]:
            metadata[f"{context}_layer{layer}"] = {
                'prompt_indices': results[context][layer]['prompt_indices'],
                'utterance_indices': results[context][layer]['utterance_indices']
            }
            break  # Only need to save indices once per context
    
    with open(BASE_PATH + 'prompt_utterance_similarity_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to prompt_utterance_similarity_metadata.json")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for context in EMOTIONS:
        if context in results:
            print(f"\n{context.upper()}:")
            layer_means = [results[context][layer]['mean_similarity'] for layer in results[context]]
            print(f"  Layers processed: {len(results[context])}")
            print(f"  Mean similarity across layers: {np.mean(layer_means):.4f}")
            print(f"  Std similarity across layers: {np.std(layer_means):.4f}")
            print(f"  Best layer: {max(results[context].items(), key=lambda x: x[1]['mean_similarity'])[0]}")
    
    print("\n" + "="*80)
#%%
# Main execution
if __name__ == "__main__":
    print("="*80)
    print("PROMPT-UTTERANCE SIMILARITY ANALYSIS")
    print("Pairwise Similarity Matrices Across All Layers")
    print("="*80)
    
    # Setup model
    model, tokenizer, device = setup_model()
    
    # Load prompt embeddings
    prompt_data, prompts_df = load_prompt_embeddings()
    
    # Extract utterance embeddings
    utterance_data, response_df = extract_utterance_embeddings(model, tokenizer, device)
    
    # Calculate similarity matrices
    results = calculate_similarity_matrices(prompt_data, utterance_data)
    
    # Save results
    save_results(results)
    
    print("\n✓ Analysis complete!")
    print("\nOutput files:")
    print("  1. prompt_utterance_similarity_matrices.npz - All similarity matrices")
    print("  2. prompt_utterance_similarity_stats.csv - Summary statistics")
    print("  3. prompt_utterance_similarity_metadata.json - Index mappings")
