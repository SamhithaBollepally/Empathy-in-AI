# %%
"""Extract per-layer hidden-state embeddings for Set 1 prompts and Set 2 utterances.

Fixes vs the previous version
-----------------------------
1. MASKED pooling. The old code did `hidden_state.mean(dim=1)` after `padding=True`,
   so padding tokens were averaged into the embedding. A short prompt batched with a
   long one got an embedding that was mostly <eos> padding, i.e. the representation
   depended on which batch the prompt happened to land in.
2. BOS EXCLUDED. In Llama the first token acts as an attention sink and its hidden
   state has a very large norm, so it dominated the mean. That is why every emotion
   prototype in the old run had cosine similarity >= 0.96 with every other one.
3. FLOAT32 ACCUMULATION. Summing up to 512 fp16 vectors loses precision; pooling is
   now done in float32.
4. LENGTH-SORTED BATCHES. Similar-length texts are batched together to cut padding
   work. Masked pooling makes this numerically identical to unsorted batching; the
   original order is restored before saving.
5. lm_head SKIPPED. Only `model.model` is run, avoiding a (B, T, 128256) logits tensor
   that is never used.
6. STACKED STORAGE. One array (num_layers, num_samples, hidden_dim) plus labels and
   texts, instead of ~116k individually-named NPZ entries.

Prompts are fed as RAW TEXT with no chat template: Set 1 measures how the model
processes emotional descriptions, not how it behaves as an assistant.

Writes to Llama-3.2-3B-Instruct/Step-6/data/ - no existing file is overwritten.
"""
import gc
import json
import os

import huggingface_hub
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# %%
token = os.environ.get('HF_TOKEN')
if token:
    huggingface_hub.login(token=token)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

# %%
MODEL_NAME = 'meta-llama/Llama-3.2-3B-Instruct'

BASE_PATH = '/Volumes/Reading/Projects/Empathy-LLM/Llama-3.2-3B-Instruct/'
# BASE_PATH = '/content/drive/MyDrive/Project-4/'
DATA_PATH = BASE_PATH + 'Step-6/data/'   # scripts live in Step-6/code-files/
os.makedirs(DATA_PATH, exist_ok=True)

BATCH_SIZE = 32      # A100-40GB comfortable; raise to 64+ on 80GB
MAX_LENGTH = 512

with open(DATA_PATH + 'dataset_stats.json') as f:
    EMOTIONS = json.load(f)['emotions']
print(f'Emotions ({len(EMOTIONS)}): {EMOTIONS}')

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map='auto',
)
model.eval()

NUM_LAYERS = model.config.num_hidden_layers + 1   # +1 for the embedding layer
HIDDEN_DIM = model.config.hidden_size
INPUT_DEVICE = next(model.parameters()).device
print(f'Model loaded. layers={NUM_LAYERS} hidden_dim={HIDDEN_DIM}')


# %%
def masked_mean_pool(hidden_state, pool_mask):
    """Mean over real (non-pad, non-BOS) tokens, accumulated in float32.

    hidden_state: (B, T, H)   pool_mask: (B, T) with 1 for tokens to include.
    returns: (B, H) float32
    """
    h = hidden_state.float()
    m = pool_mask.unsqueeze(-1).float()
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def build_pool_mask(input_ids, attention_mask):
    """attention_mask with the BOS token dropped (see fix #2)."""
    mask = attention_mask.clone()
    if tokenizer.bos_token_id is not None:
        mask[:, 0] = torch.where(input_ids[:, 0] == tokenizer.bos_token_id,
                                 torch.zeros_like(mask[:, 0]), mask[:, 0])
    # Guard: a text consisting of BOS alone would now have an all-zero mask.
    empty = mask.sum(dim=1) == 0
    if empty.any():
        mask[empty] = attention_mask[empty]
    return mask


# %%
def embed_texts(texts, batch_size=BATCH_SIZE, desc=''):
    """Per-layer masked-mean embeddings for a list of texts.

    Returns float32 array (NUM_LAYERS, len(texts), HIDDEN_DIM) in input order.
    """
    out = np.empty((NUM_LAYERS, len(texts), HIDDEN_DIM), dtype=np.float32)

    # Longest-first so the largest activation tensor is allocated on batch 1.
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    num_batches = (len(order) + batch_size - 1) // batch_size

    for b, start in enumerate(range(0, len(order), batch_size)):
        idx = order[start:start + batch_size]
        inputs = tokenizer([texts[i] for i in idx], return_tensors='pt',
                           padding=True, truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(INPUT_DEVICE) for k, v in inputs.items()}
        pool_mask = build_pool_mask(inputs['input_ids'], inputs['attention_mask'])

        with torch.no_grad():
            # model.model -> the base transformer; skips the unused lm_head projection.
            hidden_states = model.model(**inputs, output_hidden_states=True,
                                        return_dict=True).hidden_states

        for layer, hidden_state in enumerate(hidden_states):
            out[layer, idx] = masked_mean_pool(hidden_state, pool_mask).cpu().numpy()

        if (b + 1) % 20 == 0 or b + 1 == num_batches:
            print(f'  {desc} batch {b + 1}/{num_batches}')

        del hidden_states, inputs, pool_mask

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# %%
def flatten_by_emotion(texts_by_emotion):
    """{emotion: [text, ...]} -> (flat texts, labels) ordered by EMOTIONS."""
    texts, labels = [], []
    for emotion in EMOTIONS:
        items = texts_by_emotion[emotion]
        texts.extend(items)
        labels.extend([emotion] * len(items))
    return texts, np.array(labels)


# %%
# --- Set 1: prompts ---
with open(DATA_PATH + 'prompts_set1.json') as f:
    prompts_by_emotion = json.load(f)

prompt_texts, prompt_labels = flatten_by_emotion(prompts_by_emotion)
print(f'\nEmbedding {len(prompt_texts)} Set 1 prompts...')
X_prompts = embed_texts(prompt_texts, desc='prompts')
print(f'  shape: {X_prompts.shape}')

np.savez(DATA_PATH + 'prompt_embeddings.npz',
         X=X_prompts, labels=prompt_labels,
         texts=np.array(prompt_texts, dtype=object),
         emotions=np.array(EMOTIONS))
print(f'Saved: {DATA_PATH}prompt_embeddings.npz')

# %%
# --- Emotion prototypes (mean prompt embedding per emotion) + grand mean ---
# The grand mean is what makes the prototypes usable: raw prototypes share a large
# common component, so contrastive directions (proto_e - grand_mean) are what carry
# emotion-specific information. Saved here so downstream steps need not recompute it.
protos = np.stack([X_prompts[:, prompt_labels == e].mean(axis=1) for e in EMOTIONS], axis=1)
grand_mean = X_prompts.mean(axis=1)
print(f'Prototypes: {protos.shape}  grand_mean: {grand_mean.shape}')

np.savez(DATA_PATH + 'context_embeddings.npz',
         protos=protos, grand_mean=grand_mean, emotions=np.array(EMOTIONS))
print(f'Saved: {DATA_PATH}context_embeddings.npz')

del X_prompts
gc.collect()

# %%
# --- Set 2: first utterances ---
with open(DATA_PATH + 'prompts_utterances_set2.json') as f:
    set2 = json.load(f)

utterances_by_emotion = {ctx: [e['first_utterance'] for e in entries]
                         for ctx, entries in set2.items()}
utterance_texts, utterance_labels = flatten_by_emotion(utterances_by_emotion)

print(f'\nEmbedding {len(utterance_texts)} Set 2 utterances...')
X_utterances = embed_texts(utterance_texts, desc='utterances')
print(f'  shape: {X_utterances.shape}')

np.savez(DATA_PATH + 'utterance_embeddings.npz',
         X=X_utterances, labels=utterance_labels,
         texts=np.array(utterance_texts, dtype=object),
         emotions=np.array(EMOTIONS))
print(f'Saved: {DATA_PATH}utterance_embeddings.npz')

# %%
with open(DATA_PATH + 'embedding_config.json', 'w') as f:
    json.dump({
        'model': MODEL_NAME,
        'pooling': 'masked_mean_bos_excluded',
        'dtype_model': 'float16',
        'dtype_stored': 'float32',
        'prompt_format': 'raw_text_no_chat_template',
        'num_layers': NUM_LAYERS,
        'hidden_dim': HIDDEN_DIM,
        'max_length': MAX_LENGTH,
        'batch_size': BATCH_SIZE,
        'n_prompts': len(prompt_texts),
        'n_utterances': len(utterance_texts),
        'emotions': EMOTIONS,
    }, f, indent=2)
print(f'Saved: {DATA_PATH}embedding_config.json')
