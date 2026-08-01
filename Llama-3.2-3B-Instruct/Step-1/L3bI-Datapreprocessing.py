# %%
# from google.colab import drive
# drive.mount('/content/drive')

#%%
import os
import huggingface_hub
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import gc
import numpy as np
from collections import defaultdict
# import accelerate



# %%
token = os.environ.get("HF_TOKEN")
huggingface_hub.login(token=token)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# %%
model_used = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_used)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_used,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
model.eval()
print(f"Model loaded on {device}")

# %%
target_contexts = ['afraid', 'angry', 'anxious', 'devastated', 'lonely', 'sad',
                   'terrified', 'furious', 'grateful', 'hopeful', 'faithful']

BASE_PATH = '/Volumes/Reading/Projects/Empathy-LLM/Llama-3.2-3B-Instruct/'
#BASE_PATH = '/content/drive/MyDrive/Project-4/'

# %%
with open(BASE_PATH + 'prompts_set1.json', 'r') as f:
    context_prompts = json.load(f)

print("Loaded prompts_set1.json")
for ctx, prompts in context_prompts.items():
    print(f"  {ctx}: {len(prompts)} prompts")

# %%
def extract_embeddings(context_prompts, model, tokenizer, chunk_size=3, batch_size=8):
    """Extract per-layer embeddings for all prompts; compute context embeddings as mean."""
    prompt_embeddings = defaultdict(lambda: defaultdict(dict))
    context_embeddings = defaultdict(dict)
    input_device = next(model.parameters()).device

    contexts_list = list(context_prompts.keys())
    num_chunks = (len(contexts_list) + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(contexts_list))
        chunk_contexts = contexts_list[start_idx:end_idx]

        print(f"\nChunk {chunk_idx + 1}/{num_chunks}: contexts {start_idx}-{end_idx - 1}")

        chunk_prompt_embs = defaultdict(lambda: defaultdict(dict))
        chunk_context_embs = defaultdict(dict)

        for context in chunk_contexts:
            prompts = context_prompts[context]
            print(f"  {context} ({len(prompts)} prompts)")
            context_layer_embs = defaultdict(list)

            for batch_start in range(0, len(prompts), batch_size):
                batch = prompts[batch_start:batch_start + batch_size]
                if (batch_start // batch_size + 1) % 5 == 0:
                    print(f"    Batch {batch_start // batch_size + 1}/{(len(prompts) + batch_size - 1) // batch_size} done")

                inputs = tokenizer(
                    batch, return_tensors="pt",
                    padding=True, truncation=True, max_length=512
                )
                inputs = {k: v.to(input_device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True, return_dict=True)

                for layer_idx, hidden_state in enumerate(outputs.hidden_states):
                    # Mean pool over tokens per prompt in batch, then move to CPU
                    layer_embs = hidden_state.mean(dim=1).cpu().numpy()  # (batch, hidden_dim)
                    for i, prompt_idx in enumerate(range(batch_start, batch_start + len(batch))):
                        chunk_prompt_embs[context][prompt_idx][layer_idx] = layer_embs[i]
                        context_layer_embs[layer_idx].append(layer_embs[i])

            for layer_idx, embs in context_layer_embs.items():
                chunk_context_embs[context][layer_idx] = np.mean(embs, axis=0)

        for ctx in chunk_prompt_embs:
            prompt_embeddings[ctx] = chunk_prompt_embs[ctx]
        for ctx in chunk_context_embs:
            context_embeddings[ctx] = chunk_context_embs[ctx]

        del chunk_prompt_embs, chunk_context_embs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  Chunk {chunk_idx + 1} complete")

    print("\nAll embeddings extracted.")
    return prompt_embeddings, context_embeddings

# %%
def save_embeddings_npz(embeddings, filename, base_path, is_prompt=True):
    """Save embeddings dict to compressed NPZ."""
    filepath = base_path + filename
    npz_dict = {}
    if is_prompt:
        for context, prompts_dict in embeddings.items():
            for prompt_idx, layers_dict in prompts_dict.items():
                for layer_idx, emb in layers_dict.items():
                    npz_dict[f"{context}_{prompt_idx}_{layer_idx}"] = emb
    else:
        for context, layers_dict in embeddings.items():
            for layer_idx, emb in layers_dict.items():
                npz_dict[f"{context}_{layer_idx}"] = emb
    np.savez_compressed(filepath, **npz_dict)
    print(f"Saved: {filepath}")

# %%
CHUNK_SIZE = 3
BATCH_SIZE = 8
prompt_embeddings, context_embeddings = extract_embeddings(
    context_prompts, model, tokenizer, CHUNK_SIZE, BATCH_SIZE
)

# %%
print("Saving embeddings to NPZ...")
save_embeddings_npz(prompt_embeddings, 'prompt_embeddings.npz', BASE_PATH + 'Step-1/', is_prompt=True)
save_embeddings_npz(context_embeddings, 'context_embeddings.npz', BASE_PATH + 'Step-1/', is_prompt=False)

first_ctx = next(iter(prompt_embeddings))
first_prompt = next(iter(prompt_embeddings[first_ctx]))
num_layers = len(prompt_embeddings[first_ctx][first_prompt])
print(f"Layers per prompt: {num_layers}")

# %%
with open(BASE_PATH + 'prompts_utterances_set2.json', 'r') as f:
    prompts_utterances = json.load(f)

# Extract only the first_utterance texts, keyed by context
context_utterances = {
    ctx: [entry['first_utterance'] for entry in entries]
    for ctx, entries in prompts_utterances.items()
}

print("Loaded prompts_utterances_set2.json")
for ctx, utts in context_utterances.items():
    print(f"  {ctx}: {len(utts)} utterances")

# %%
utterance_embeddings, _ = extract_embeddings(
    context_utterances, model, tokenizer, CHUNK_SIZE, BATCH_SIZE
)

# %%
print("Saving utterance embeddings to NPZ...")
save_embeddings_npz(utterance_embeddings, 'utterance_embeddings.npz', BASE_PATH + 'Step-1/', is_prompt=True)
print(f"Layers per utterance: {len(utterance_embeddings[first_ctx][0])}")
