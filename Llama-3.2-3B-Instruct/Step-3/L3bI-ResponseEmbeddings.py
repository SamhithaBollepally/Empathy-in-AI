
#%%
# from google.colab import drive
# drive.mount('/content/drive')

# %%
import os
import json
import gc
import getpass
import numpy as np
import torch
import huggingface_hub
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

# %%
BASE_PATH = '/Volumes/Reading/Projects/Empathy-LLM/Llama-3.2-3B-Instruct/'
#BASE_PATH = '/content/drive/MyDrive/Project-4/'

OUTPUTS_PATH = BASE_PATH + 'Outputs/'
os.makedirs(OUTPUTS_PATH, exist_ok=True)

# (input response file, output embeddings file)
RESPONSE_FILES = [
    ('CoT_responses.json',      'CoT_response_embeddings.npz'),
    ('zeroshot_responses.json', 'zeroshot_response_embeddings.npz'),
]

# %%
token = getpass.getpass("Enter your Hugging Face token: ")
huggingface_hub.login(token=token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# %%
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
model.eval()
MODEL_DEVICE = next(model.parameters()).device  # compute once
print("Model loaded.")

# %%
BATCH_SIZE = 8
MAX_LENGTH = 512

def masked_mean_pool(hidden_state, attention_mask):
    """Mean-pool token embeddings using the attention mask (ignores padding).
    hidden_state: (B, T, H) float tensor
    attention_mask: (B, T) int/float tensor
    returns: (B, H) float tensor
    """
    mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)  # (B, T, 1)
    summed = (hidden_state * mask).sum(dim=1)                    # (B, H)
    counts = mask.sum(dim=1).clamp(min=1)                        # (B, 1) avoid div-by-zero
    return summed / counts

def extract_response_embeddings(data):
    """For every response, return per-layer mean-pooled embeddings.

    Mirrors the prompt-embedding method: forward each text with
    output_hidden_states=True, then mean-pool each layer over tokens.

    Returns: {emotion: {response_idx: {layer_idx: np.ndarray(H,)}}}
    Empty responses are skipped (no meaningful text to embed).
    """
    embeddings = defaultdict(lambda: defaultdict(dict))

    for emotion, items in data.items():
        # Keep original indices so keys stay aligned with the source file.
        indexed = [(i, it.get('response', '') or '') for i, it in enumerate(items)]
        indexed = [(i, r) for i, r in indexed if r.strip()]
        n_skipped = len(items) - len(indexed)
        print(f"  {emotion}: {len(indexed)} responses"
              + (f" ({n_skipped} empty skipped)" if n_skipped else ""))

        for start in range(0, len(indexed), BATCH_SIZE):
            batch = indexed[start:start + BATCH_SIZE]
            batch_idxs = [i for i, _ in batch]
            batch_texts = [r for _, r in batch]

            inputs = tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH
            ).to(MODEL_DEVICE)

            with torch.inference_mode():
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)

            attn = inputs['attention_mask']
            for layer_idx, hidden_state in enumerate(outputs.hidden_states):
                pooled = masked_mean_pool(hidden_state, attn)  # (B, H)
                pooled = pooled.float().cpu().numpy()
                for row, resp_idx in enumerate(batch_idxs):
                    embeddings[emotion][resp_idx][layer_idx] = pooled[row]

            del inputs, outputs, attn
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return embeddings

def save_embeddings_to_npz(embeddings, filepath):
    """Flatten {emotion: {idx: {layer: emb}}} to keys 'emotion_idx_layer'."""
    npz_dict = {}
    for emotion, idx_dict in embeddings.items():
        for resp_idx, layer_dict in idx_dict.items():
            for layer_idx, emb in layer_dict.items():
                npz_dict[f"{emotion}_{resp_idx}_{layer_idx}"] = emb
    np.savez_compressed(filepath, **npz_dict)
    print(f"Saved {len(npz_dict)} vectors to {filepath}")

# %%
for in_name, out_name in RESPONSE_FILES:
    in_path = OUTPUTS_PATH + in_name
    out_path = OUTPUTS_PATH + out_name
    print(f"\n=== Embedding responses from {in_name} ===")

    with open(in_path) as f:
        data = json.load(f)

    embeddings = extract_response_embeddings(data)
    save_embeddings_to_npz(embeddings, out_path)

    # Report layer count from the first available response.
    first_emotion = next(iter(embeddings))
    first_idx = next(iter(embeddings[first_emotion]))
    num_layers = len(embeddings[first_emotion][first_idx])
    dim = embeddings[first_emotion][first_idx][0].shape[0]
    print(f"Layers per response: {num_layers} | embedding dim: {dim}")

    del embeddings, data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\nDone. Response embeddings saved for both files.")
