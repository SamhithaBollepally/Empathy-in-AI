import pandas as pd
import numpy as np
import json
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import random
import gc
import os

hf_token = os.environ.get("HF_TOKEN")
# Login to Hugging Face
login(token=hf_token)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
tokenizer.pad_token = tokenizer.eos_token

# Use half-precision to save memory
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    dtype=torch.float16,
    low_cpu_mem_usage=True)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Move model to GPU if available
model = model.to(device)
print(f"Model moved to {device}")

# Base path for Google Drive
BASE_PATH = '/content/drive/MyDrive/Project-4/'

def extract_embeddings(context_prompts, model, tokenizer, device, chunk_size=3):
    """Extract embeddings for all prompts and aggregate by context."""
    prompt_embeddings = defaultdict(lambda: defaultdict(dict))
    context_embeddings = defaultdict(dict)
    
    contexts_list = list(context_prompts.keys())
    num_chunks = (len(contexts_list) + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(contexts_list))
        chunk_contexts = contexts_list[start_idx:end_idx]
        
        print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks}: contexts {start_idx}-{end_idx-1}")
        
        chunk_prompt_embeddings = defaultdict(lambda: defaultdict(dict))
        chunk_context_embeddings = defaultdict(dict)
        
        for context in chunk_contexts:
            prompts = context_prompts[context]
            print(f"  Processing context: {context} ({len(prompts)} prompts)")
            
            context_layer_embeddings = defaultdict(list)
            
            for prompt_idx, prompt in enumerate(prompts):
                if (prompt_idx + 1) % 20 == 0:
                    print(f"    Processed {prompt_idx + 1}/{len(prompts)} prompts")
                
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}  # Move to GPU
                
                # Model inference on GPU (70-80% of total time)
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                
                # Extract embeddings from all layers
                for layer_idx, hidden_state in enumerate(outputs.hidden_states):
                    # Mean pooling happens on GPU, then move to CPU for storage
                    layer_embedding = hidden_state.mean(dim=1).squeeze().cpu().numpy()
                    chunk_prompt_embeddings[context][prompt_idx][layer_idx] = layer_embedding
                    context_layer_embeddings[layer_idx].append(layer_embedding)
            
            for layer_idx, embeddings_list in context_layer_embeddings.items():
                chunk_context_embeddings[context][layer_idx] = np.mean(embeddings_list, axis=0)
        
        for context in chunk_prompt_embeddings:
            prompt_embeddings[context] = chunk_prompt_embeddings[context]
        for context in chunk_context_embeddings:
            context_embeddings[context] = chunk_context_embeddings[context]
        
        del chunk_prompt_embeddings, chunk_context_embeddings
        gc.collect()
        
        print(f"  Chunk {chunk_idx + 1} complete")
    
    print("\n✓ All embedding extraction complete!")
    return prompt_embeddings, context_embeddings


def save_embeddings_to_npz(embeddings, filename, base_path, is_prompt=True):
    """Save embeddings to compressed NPZ file (10-100x faster than JSON)."""
    filepath = base_path + filename
    
    if is_prompt:
        # Flatten nested dict: context -> prompt_idx -> layer -> embedding
        # Save as: "context_promptidx_layer": embedding_array
        npz_dict = {}
        for context, prompts_dict in embeddings.items():
            for prompt_idx, layers_dict in prompts_dict.items():
                for layer_idx, emb in layers_dict.items():
                    key = f"{context}_{prompt_idx}_{layer_idx}"
                    npz_dict[key] = emb
        np.savez_compressed(filepath, **npz_dict)
    else:
        # Context embeddings: context -> layer -> embedding
        # Save as: "context_layer": embedding_array
        npz_dict = {}
        for context, layers_dict in embeddings.items():
            for layer_idx, emb in layers_dict.items():
                key = f"{context}_{layer_idx}"
                npz_dict[key] = emb
        np.savez_compressed(filepath, **npz_dict)
    
    print(f"Saved {filepath}")


with open(BASE_PATH + 'context_prompts.json', 'r') as f:
    context_prompts = json.load(f)

CHUNK_SIZE = 3
prompt_embeddings, context_embeddings = extract_embeddings(
    context_prompts, model, tokenizer, device, CHUNK_SIZE
)

print("\nSaving embeddings to NPZ format (fast & compressed)...")
save_embeddings_to_npz(prompt_embeddings, 'prompt_embeddings.npz', BASE_PATH, is_prompt=True)
save_embeddings_to_npz(context_embeddings, 'context_embeddings.npz', BASE_PATH, is_prompt=False)

# Get number of layers from first context and first prompt
first_context = next(iter(prompt_embeddings.keys()))
first_prompt = next(iter(prompt_embeddings[first_context].keys()))
num_layers = len(prompt_embeddings[first_context][first_prompt])
print(f"Layers per prompt: {num_layers}")

def compute_cosine_similarities(prompt_embeddings, context_embeddings, context_prompts, num_layers, chunk_size=3):
    """Compute cosine similarity between prototype and prompt embeddings."""
    cosine_similarities = defaultdict(dict)
    contexts_list = list(context_prompts.keys())
    num_chunks = (len(contexts_list) + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(contexts_list))
        chunk_contexts = contexts_list[start_idx:end_idx]
        
        print(f"\nProcessing cosine similarity chunk {chunk_idx + 1}/{num_chunks}")
        
        for context in chunk_contexts:
            print(f"  Computing similarities for context: {context}")
            
            for layer_idx in range(num_layers):
                context_emb = context_embeddings[context][layer_idx].reshape(1, -1)
                prompt_sims = []
                
                # Cosine similarity on CPU (sklearn doesn't support GPU)
                for prompt_idx in range(len(context_prompts[context])):
                    prompt_emb = prompt_embeddings[context][prompt_idx][layer_idx].reshape(1, -1)
                    sim = cosine_similarity(context_emb, prompt_emb)[0][0]
                    prompt_sims.append(sim)
                
                cosine_similarities[context][layer_idx] = prompt_sims
    
    return cosine_similarities


cosine_similarities = compute_cosine_similarities(
    prompt_embeddings, context_embeddings, context_prompts, num_layers, CHUNK_SIZE
)

# Save cosine similarities as NPZ (faster)
cos_sim_npz = {f"{context}_{layer}": np.array(sims) 
               for context, layers_dict in cosine_similarities.items() 
               for layer, sims in layers_dict.items()}
filepath = BASE_PATH + 'cosine_similarities.npz'
np.savez_compressed(filepath, **cos_sim_npz)
print(f"Saved cosine similarities to {filepath}")

def train_probing_classifier(prompt_embeddings, context_prompts, num_layers, layer_chunk_size=5):
    """Train logistic regression probing classifiers per layer (optimized)."""
    layer_accuracies = {}
    layer_accuracies_shuffled = {}
    context_layer_accuracies = defaultdict(dict)
    context_layer_accuracies_shuffled = defaultdict(dict)
    
    # Optimization 1: Prepare data once for all layers
    print("\nPreparing data for all layers...")
    X_all_layers = []  # Will be shape: (num_samples, num_layers, embedding_dim)
    y_all = []
    context_list = list(context_prompts.keys())
    
    for context in context_list:
        for prompt_idx in range(len(context_prompts[context])):
            # Stack embeddings from all layers for this prompt
            layer_embs = [prompt_embeddings[context][prompt_idx][layer_idx] 
                         for layer_idx in range(num_layers)]
            X_all_layers.append(layer_embs)
            y_all.append(context)
    
    X_all_layers = np.array(X_all_layers)  # Shape: (num_samples, num_layers, embedding_dim)
    y_all = np.array(y_all)
    print(f"Data prepared: {X_all_layers.shape}, Labels: {len(y_all)}")
    
    # Optimization 4: Split indices once, reuse for all layers
    indices = np.arange(len(y_all))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_all)
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]
    
    num_layer_chunks = (num_layers + layer_chunk_size - 1) // layer_chunk_size
    
    for layer_chunk_idx in range(num_layer_chunks):
        start_layer = layer_chunk_idx * layer_chunk_size
        end_layer = min(start_layer + layer_chunk_size, num_layers)
        
        for layer_idx in range(start_layer, end_layer):
            print(f"\n--- Layer {layer_idx} ---")
            
            # Fast array slice instead of rebuilding
            X = X_all_layers[:, layer_idx, :]
            X_train = X[train_idx]
            X_test = X[test_idx]
            
            # Logistic Regression on CPU (sklearn doesn't support GPU)
            clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            layer_accuracies[layer_idx] = accuracy
            print(f"Real labels accuracy: {accuracy:.4f}")
            
            # Optimization 2: Vectorized per-context accuracy using confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=context_list)
            per_context_acc = cm.diagonal() / cm.sum(axis=1)
            for i, context in enumerate(context_list):
                context_layer_accuracies[context][layer_idx] = per_context_acc[i]
            
            # Shuffled baseline
            y_train_shuffled = y_train.copy()
            random.shuffle(y_train_shuffled)
            clf_shuffled = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
            clf_shuffled.fit(X_train, y_train_shuffled)
            y_pred_shuffled = clf_shuffled.predict(X_test)
            accuracy_shuffled = accuracy_score(y_test, y_pred_shuffled)
            layer_accuracies_shuffled[layer_idx] = accuracy_shuffled
            print(f"Shuffled labels accuracy: {accuracy_shuffled:.4f}")
            
            # Vectorized per-context accuracy for shuffled
            cm_shuffled = confusion_matrix(y_test, y_pred_shuffled, labels=context_list)
            per_context_acc_shuffled = cm_shuffled.diagonal() / cm_shuffled.sum(axis=1)
            for i, context in enumerate(context_list):
                context_layer_accuracies_shuffled[context][layer_idx] = per_context_acc_shuffled[i]
            
            del X, X_train, X_test, clf, clf_shuffled, y_pred, y_pred_shuffled
            gc.collect()
    
    # Clean up large array
    del X_all_layers
    gc.collect()
    
    return layer_accuracies, layer_accuracies_shuffled, context_layer_accuracies, context_layer_accuracies_shuffled


LAYER_CHUNK_SIZE = 5
layer_accuracies, layer_accuracies_shuffled, context_layer_accuracies, context_layer_accuracies_shuffled = train_probing_classifier(
    prompt_embeddings, context_prompts, num_layers, LAYER_CHUNK_SIZE
)


def find_best_layers(layer_accuracies, layer_accuracies_shuffled, context_layer_accuracies, 
                     context_layer_accuracies_shuffled, context_prompts):
    """Find best performing layers overall and per context."""
    best_layers_per_context = {}
    for context in context_prompts.keys():
        best_layer = max(context_layer_accuracies[context].items(), key=lambda x: x[1])
        best_layers_per_context[context] = {"layer": best_layer[0], "accuracy": best_layer[1]}
        print(f"Original: {context}: Layer {best_layer[0]} (accuracy: {best_layer[1]:.4f})")
    
    best_layers_per_context_shuffled = {}
    for context in context_prompts.keys():
        best_layer = max(context_layer_accuracies_shuffled[context].items(), key=lambda x: x[1])
        best_layers_per_context_shuffled[context] = {"layer": best_layer[0], "accuracy": best_layer[1]}
        print(f"Shuffled:{context}: Layer {best_layer[0]} (accuracy: {best_layer[1]:.4f})")
    
    best_layer_overall = max(layer_accuracies.items(), key=lambda x: x[1])
    print(f"Layer {best_layer_overall[0]}: {best_layer_overall[1]:.4f}")
    
    best_layer_overall_shuffled = max(layer_accuracies_shuffled.items(), key=lambda x: x[1])
    print(f"Layer {best_layer_overall_shuffled[0]}: {best_layer_overall_shuffled[1]:.4f}")
    
    return best_layers_per_context, best_layers_per_context_shuffled, best_layer_overall, best_layer_overall_shuffled


best_layers_per_context, best_layers_per_context_shuffled, best_layer_overall, best_layer_overall_shuffled = find_best_layers(
    layer_accuracies, layer_accuracies_shuffled, context_layer_accuracies, 
    context_layer_accuracies_shuffled, context_prompts
)

# Save probing results as JSON (human-readable for final analysis)
probing_results = {
    "layer_accuracies": {str(k): v for k, v in layer_accuracies.items()},
    "layer_accuracies_shuffled": {str(k): v for k, v in layer_accuracies_shuffled.items()},
    "context_layer_accuracies": {k: {str(layer): acc for layer, acc in v.items()} 
                                 for k, v in context_layer_accuracies.items()},
    "context_layer_accuracies_shuffled": {k: {str(layer): acc for layer, acc in v.items()} 
                                          for k, v in context_layer_accuracies_shuffled.items()},
    "best_layers_per_context": {k: {"layer": str(v["layer"]), "accuracy": v["accuracy"]} 
                               for k, v in best_layers_per_context.items()},
    "best_layers_per_context_shuffled": {k: {"layer": str(v["layer"]), "accuracy": v["accuracy"]} 
                                        for k, v in best_layers_per_context_shuffled.items()},
    "best_layer_overall": {"layer": str(best_layer_overall[0]), "accuracy": best_layer_overall[1]},
    "best_layer_overall_shuffled": {"layer": str(best_layer_overall_shuffled[0]), "accuracy": best_layer_overall_shuffled[1]}
}

filepath = BASE_PATH + 'probing_results.json'
with open(filepath, 'w') as f:
    json.dump(probing_results, f, indent=2)
print(f"\nSaved probing results to {filepath}")