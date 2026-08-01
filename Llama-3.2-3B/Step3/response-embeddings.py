# %% [markdown]
# # Response Generation and Embedding Extraction

# %% Imports
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import json
import numpy as np
import os
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# %% Configuration
BASE_PATH = '/content/drive/MyDrive/Project-4/'
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_NAME = "meta-llama/Llama-3.2-3B"

BEST_LAYERS = {
    'afraid': 6, 'angry': 5, 'anxious': 9, 'devastated': 8,
    'lonely': 4, 'sad': 7, 'terrified': 1, 'furious': 7,
    'grateful': 15, 'hopeful': 12, 'faithful': 5
}

# %% Setup Functions
def setup_model():
    """Load and configure model on GPU"""
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
    
    return model, tokenizer, device

def generate_response(utterance, model, tokenizer, device):
    """Generate empathetic response"""
    prompt = f"Patient: {utterance}\nMedical Assistant:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, do_sample=True, temperature=0.7, top_p=0.9,
            max_new_tokens=50, pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("Medical Assistant:")[-1].strip()
    
    if '.' in response:
        response = response.split('.')[0] + '.'
    
    return response

def extract_embeddings(text, model, tokenizer, device):
    """Extract embeddings from all layers"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    
    embeddings = {}
    for layer_idx, hidden_state in enumerate(outputs.hidden_states):
        embeddings[layer_idx] = hidden_state.mean(dim=1).squeeze().cpu().numpy()
    
    return embeddings

def process_data(df, model, tokenizer, device):
    """Generate responses and extract embeddings"""
    responses = []
    all_embeddings = defaultdict(lambda: defaultdict(dict))
    best_embeddings = {}
    
    for idx, row in df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df)}")
        
        context = row['context']
        
        # Generate response
        response = generate_response(row['first_utterance'], model, tokenizer, device)
        
        # Extract embeddings
        embeddings = extract_embeddings(response, model, tokenizer, device)
        
        # Store all layers
        for layer_idx, emb in embeddings.items():
            all_embeddings[context][idx][layer_idx] = emb
        
        # Store best layer
        best_layer = BEST_LAYERS[context]
        best_embeddings[f"{context}_{idx}"] = embeddings[best_layer]
        
        responses.append({
            'context': context,
            'prompt': row['prompt'],
            'utterance': row['first_utterance'],
            'response': response
        })
    
    return responses, all_embeddings, best_embeddings

def save_outputs(responses, all_embeddings, best_embeddings):
    """Save all outputs to files"""
    # Save responses
    pd.DataFrame(responses).to_csv(BASE_PATH + 'generated_responses.csv', index=False)
    
    with open(BASE_PATH + 'generated_responses.json', 'w') as f:
        json.dump(responses, f, indent=2)
    
    # Save all layer embeddings
    npz_dict = {}
    for context, response_dict in all_embeddings.items():
        for response_idx, layers_dict in response_dict.items():
            for layer_idx, emb in layers_dict.items():
                npz_dict[f"{context}_{response_idx}_{layer_idx}"] = emb
    
    np.savez_compressed(BASE_PATH + 'response_embeddings.npz', **npz_dict)
    np.savez_compressed(BASE_PATH + 'response_embeddings_best_layer.npz', **best_embeddings)
    
    print(f"\n✓ Generated {len(responses)} responses")
    print(f"✓ Saved to {BASE_PATH}generated_responses.csv/.json")
    print(f"✓ Saved embeddings to response_embeddings.npz and response_embeddings_best_layer.npz")

def analyze_similarity(best_embeddings):
    """Analyze cosine similarity with prototypes"""
    print("\n" + "="*60)
    print("COSINE SIMILARITY ANALYSIS")
    print("="*60)
    
    context_embs = np.load(BASE_PATH + 'context_embeddings.npz')
    emotions = list(BEST_LAYERS.keys())
    results = []
    
    for key, response_emb in best_embeddings.items():
        context = key.rsplit('_', 1)[0]
        best_layer = BEST_LAYERS[context]
        response_emb = response_emb.reshape(1, -1)
        
        # Compare against all emotion prototypes
        sims = {}
        for emotion in emotions:
            proto = context_embs[f"{emotion}_{best_layer}"].reshape(1, -1)
            sims[emotion] = cosine_similarity(response_emb, proto)[0][0]
        
        predicted_emotion = max(sims, key=sims.get)
        
        results.append({
            'context': context,
            'correct_sim': sims[context],
            'predicted_emotion': predicted_emotion,
            'correct': (predicted_emotion == context),
            **{f"sim_{e}": sims[e] for e in emotions}
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(BASE_PATH + 'response_similarity_analysis.csv', index=False)
    
    # Print statistics
    accuracy_per_context = results_df.groupby('context')['correct'].mean()
    print("\nAccuracy per context:")
    for context in emotions:
        print(f"  {context:<12}: {accuracy_per_context[context]:.2%}")
    
    print(f"\nOverall accuracy: {results_df['correct'].mean():.2%}")
    print(f"✓ Saved to {BASE_PATH}response_similarity_analysis.csv")
    print("="*60)
    
    return results_df

# %% Main Execution
if __name__ == "__main__":
    # Setup
    model, tokenizer, device = setup_model()
    df = pd.read_csv(BASE_PATH + 'response_prompts.csv')
    torch.manual_seed(42)
    
    print(f"Processing {len(df)} utterances...\n")
    
    # Generate and extract
    responses, all_embeddings, best_embeddings = process_data(df, model, tokenizer, device)
    
    # Save outputs
    save_outputs(responses, all_embeddings, best_embeddings)
    
    # Analyze similarity
    results_df = analyze_similarity(best_embeddings)
