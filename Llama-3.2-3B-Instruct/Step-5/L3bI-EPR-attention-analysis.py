# %%
# ============================================================================
# Step 5j — Attention-Flow Analysis
#
# For each generated response, extracts per-token attention weights and answers:
#   1. Which tokens does the model attend to when generating the response?
#   2. How much attention goes to prompt tokens vs already-generated tokens?
#   3. Does the model attend more to emotionally relevant prompt tokens?
#      (detected via the L15 emotion probe, not a word list)
#   4. How does attention change across layers (layer-wise flow)?
#
# Run on any saved response file (baseline or steered) and compare summaries.
# ============================================================================

# %%
import os
import gc
import json
import getpass
import numpy as np
import torch
import huggingface_hub
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

# %%
# ── Config ───────────────────────────────────────────────────────────────────
BASE_PATH = '/Volumes/Reading/Projects/Empathy-LLM/Llama-3.2-3B-Instruct/'
# BASE_PATH = '/content/drive/MyDrive/Project-4/'

STEP5_PATH = BASE_PATH + 'Step-5/'

# File to analyse — change to any saved response file
INPUT_FILE = STEP5_PATH + 'epr_transform_cubic_a0.0001_l1.json'
# INPUT_FILE = STEP5_PATH + 'zeroshot_responses.json'

OUTPUT_FILE = STEP5_PATH + 'attention_analysis_' + os.path.basename(INPUT_FILE)

N_SAMPLES_PER_EMOTION  = 5    # samples per emotion to analyse
ANALYSIS_MAX_NEW_TOKENS = 50  # token budget per sample

# %%
# ── Authentication and model load ────────────────────────────────────────────
token = getpass.getpass("HuggingFace token: ")
huggingface_hub.login(token=token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = 'left'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    attn_implementation="eager",  # required to return attention weights
)
model.eval()
MODEL_DEVICE = next(model.parameters()).device
print(f"Model loaded on {MODEL_DEVICE}")

# %%
# ── Load emotion probe for identifying emotionally relevant prompt tokens ─────
# Colab layout: set STEP2_PATH = BASE_PATH
STEP2_PATH = BASE_PATH + 'Step-2/'
bundle = torch.load(STEP2_PATH + 'emotion_probes.pt', map_location='cpu', weights_only=False)
EMOTION_CLASSES = [str(c) for c in bundle['classes']]
CLASS_TO_IDX    = {c: i for i, c in enumerate(EMOTION_CLASSES)}

PROBE_LAYER = 15   # layer with highest probing accuracy
probe_scaler = bundle['layers'][PROBE_LAYER]['scaler']
probe_lin = torch.nn.Linear(bundle['hidden_dim'], len(EMOTION_CLASSES), bias=True)
probe_lin.load_state_dict(bundle['layers'][PROBE_LAYER]['l2_state_dict'])
probe_lin.eval()

probe_W     = probe_lin.weight.detach().to(torch.float32)
probe_b     = probe_lin.bias.detach().to(torch.float32)
probe_mean  = torch.tensor(probe_scaler.mean_,  dtype=torch.float32)
probe_scale = torch.tensor(probe_scaler.scale_, dtype=torch.float32)
print(f"Loaded emotion probe for layer {PROBE_LAYER}")

# %%
# ── Helpers ───────────────────────────────────────────────────────────────────
GEN_SYSTEM_PROMPT = (
    "You are a medical assistant. Given the user's emotional situation, do two things:\n"
    "1. Write a response (1-2 sentences).\n"
    "2. Rate how emotional is the user's situation on a scale of 1 (mild) to 5 (highly emotional).\n\n"
    "Format your output exactly as:\n"
    "Response: <your response>\n"
    "Intensity: <1-5>"
)

def format_chat(system, user):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def emotional_token_positions(h15, emotion):
    """
    h15    : (prompt_len, 3072) hidden state at PROBE_LAYER for prompt tokens
    emotion: target emotion string
    Returns: positions where the probe's top prediction == target emotion
    """
    target_idx = CLASS_TO_IDX.get(emotion)
    if target_idx is None or h15 is None or h15.shape[0] == 0:
        return []
    h      = h15.float()
    scaled = (h - probe_mean.to(h.device)) / probe_scale.to(h.device)
    scores = scaled @ probe_W.to(h.device).T + probe_b.to(h.device)  # (prompt_len, 11)
    pred   = scores.argmax(dim=-1).cpu().tolist()
    return [pos for pos, p in enumerate(pred) if p == target_idx]

def sample_token(logits, temperature=0.7, top_p=0.9):
    """Sample one token using temperature + nucleus (top-p) filtering."""
    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum > top_p
    mask[:, 0] = False      # keep at least one token
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True) + 1e-8
    samples = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices.gather(1, samples)

# %%
# ── Attention extraction for one sample ──────────────────────────────────────
def analyse_one(utterance, emotion, max_new_tokens=50):
    """
    Manual token-by-token generation with use_cache=True and output_attentions=True.
    Stops at EOS. Returns per-token and per-layer attention statistics.
    """
    chat   = format_chat(GEN_SYSTEM_PROMPT, utterance)
    inputs = tokenizer(chat, return_tensors='pt').to(MODEL_DEVICE)
    prompt_len     = inputs['input_ids'].shape[1]
    input_ids      = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    generated_ids       = []
    captured_attentions = []   # list[list[Tensor(1, heads, seq)]] per generated token

    # ── Prefill ──────────────────────────────────────────────────────────────
    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
        )
    # Attention for the first generated token: last query row of each layer
    # out.attentions: tuple of (1, heads, prompt_len, prompt_len) per layer
    captured_attentions.append(
        [a[0, :, -1, :].unsqueeze(0).cpu() for a in out.attentions]
    )
    # h15 for probe-based emotional token detection
    h15 = out.hidden_states[PROBE_LAYER][0, :prompt_len, :].detach()
    past_key_values = out.past_key_values

    logits = out.logits[:, -1, :]
    next_token = sample_token(logits)
    generated_ids.append(next_token.item())

    # ── Autoregressive generation ─────────────────────────────────────────────
    for _ in range(1, max_new_tokens):
        if next_token.item() == tokenizer.eos_token_id:
            break
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=torch.long, device=MODEL_DEVICE)],
            dim=1
        )
        with torch.inference_mode():
            out = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
            )
        # out.attentions: tuple of (1, heads, 1, prompt_len + t) per layer
        captured_attentions.append(
            [a[0, :, 0, :].unsqueeze(0).cpu() for a in out.attentions]
        )
        past_key_values = out.past_key_values
        logits = out.logits[:, -1, :]
        next_token = sample_token(logits)
        generated_ids.append(next_token.item())

    # ── Decode ────────────────────────────────────────────────────────────────
    full_ids         = inputs['input_ids'][0].cpu().tolist() + generated_ids
    all_tokens       = [tokenizer.decode([tid], skip_special_tokens=False) for tid in full_ids]
    prompt_tokens    = all_tokens[:prompt_len]
    generated_tokens = all_tokens[prompt_len:]
    emo_positions    = emotional_token_positions(h15, emotion)
    gen_len          = len(generated_ids)
    num_layers       = len(captured_attentions[0])

    del inputs, out, past_key_values, h15
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Per-token stats ───────────────────────────────────────────────────────
    step_stats = []
    for t in range(gen_len):
        step_attns  = captured_attentions[t]
        # Each tensor shape: (1, heads, cur_seq_len)
        cur_seq_len = step_attns[0].shape[-1]

        stacked = torch.stack(step_attns)          # (num_layers, 1, heads, cur_seq_len)
        avg     = stacked.mean(dim=(0, 1, 2)).numpy()  # (cur_seq_len,)

        prompt_attn  = float(avg[:prompt_len].sum())
        gen_attn     = float(avg[prompt_len:].sum())
        emo_attn     = float(avg[emo_positions].sum()) if emo_positions else 0.0
        max_pos      = int(np.argmax(avg))
        max_token    = all_tokens[max_pos] if max_pos < len(all_tokens) else ''

        # Per-layer: average over heads → (num_layers, cur_seq_len)
        layer_avg    = stacked.mean(dim=(1, 2)).numpy()
        layer_prompt = layer_avg[:, :prompt_len].sum(axis=1)
        layer_gen    = layer_avg[:, prompt_len:].sum(axis=1)
        layer_emo    = (
            layer_avg[:, emo_positions].sum(axis=1) if emo_positions
            else np.zeros(num_layers)
        )

        # Per-head per-layer: (num_layers, heads, cur_seq_len)
        # stacked shape: (num_layers, 1, heads, cur_seq_len)
        per_head = stacked[:, 0, :, :].numpy()   # (num_layers, heads, cur_seq_len)
        num_heads = per_head.shape[1]

        # Emotional attention per head per layer: (num_layers, heads)
        if emo_positions:
            head_emo = per_head[:, :, emo_positions].sum(axis=2)   # (num_layers, heads)
        else:
            head_emo = np.zeros((num_layers, num_heads))

        # Head variance at each layer: how much do heads disagree on emotional tokens?
        # var across 24 heads → (num_layers,)
        head_emo_variance = head_emo.var(axis=1)   # (num_layers,)

        # Which head attends most to emotional tokens at each layer? → (num_layers,)
        top_emo_head = head_emo.argmax(axis=1).tolist()

        # Mean emotional attention per head averaged across layers → (heads,)
        mean_head_emo = head_emo.mean(axis=0)   # (heads,)

        step_stats.append({
            'step':                    t,
            'generated_token':         generated_tokens[t],
            'prompt_attention':        round(prompt_attn, 4),
            'generated_attention':     round(gen_attn, 4),
            'emotional_attention':     round(emo_attn, 4),
            'max_attention_token':     max_token,
            'max_attention_is_prompt': max_pos < prompt_len,
            'layerwise': {
                'prompt_attention':    [round(float(x), 4) for x in layer_prompt],
                'generated_attention': [round(float(x), 4) for x in layer_gen],
                'emotional_attention': [round(float(x), 4) for x in layer_emo],
                # per-head emotional attention: list[layer] of list[head]
                'head_emo_attention':  [[round(float(v), 4) for v in head_emo[L]]
                                        for L in range(num_layers)],
                # variance across heads at each layer
                'head_emo_variance':   [round(float(v), 6) for v in head_emo_variance],
                # which head attended most to emotional tokens at each layer
                'top_emo_head':        top_emo_head,
            },
            # mean emotional attention per head (averaged across all layers)
            'mean_head_emo': [round(float(v), 4) for v in mean_head_emo],
        })

    return {
        'emotion':             emotion,
        'utterance':           utterance,
        'prompt_len':          prompt_len,
        'gen_len':             gen_len,
        'num_layers':          num_layers,
        'n_emotional_tokens':  len(emo_positions),
        'prompt_tokens':       prompt_tokens,
        'generated_tokens':    generated_tokens,
        'emotional_positions': emo_positions,
        'step_stats':          step_stats,
    }

# %%
# ── Load responses and build sample list ─────────────────────────────────────
with open(INPUT_FILE) as f:
    data = json.load(f)

samples = []
for emotion in list(data.keys()):
    for it in data[emotion][:N_SAMPLES_PER_EMOTION]:
        samples.append((emotion, it['utterance']))

print(f"Analysing {len(samples)} samples from:\n  {INPUT_FILE}")

# %%
# ── Run analysis sample by sample ────────────────────────────────────────────
per_sample = []
for idx, (emotion, utterance) in enumerate(samples):
    print(f"  [{idx+1}/{len(samples)}] {emotion} — {utterance[:60]}...")
    result = analyse_one(utterance, emotion, max_new_tokens=ANALYSIS_MAX_NEW_TOKENS)
    per_sample.append(result)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# %%
# ── Aggregate statistics ──────────────────────────────────────────────────────
def _mean(arr):
    return round(float(np.mean(arr)), 4) if arr else None

def _last_n_mean(arr, n=10):
    return round(float(np.mean(arr[-n:])), 4) if len(arr) >= n else _mean(arr)

overall = defaultdict(list)
per_emotion = defaultdict(lambda: defaultdict(list))

n_layers = per_sample[0]['num_layers'] if per_sample else 28
n_heads  = len(per_sample[0]['step_stats'][0]['mean_head_emo']) if per_sample else 24

# Accumulators: layer-wise and head-wise
lw_prompt       = np.zeros(n_layers)
lw_gen          = np.zeros(n_layers)
lw_emo          = np.zeros(n_layers)
lw_head_emo     = np.zeros((n_layers, n_heads))   # (layers, heads)
lw_head_var     = np.zeros(n_layers)              # variance across heads per layer
head_emo_global = np.zeros(n_heads)               # mean per head across all layers & tokens
lw_count        = 0

for r in per_sample:
    e = r['emotion']
    for step in r['step_stats']:
        overall['prompt'].append(step['prompt_attention'])
        overall['generated'].append(step['generated_attention'])
        overall['emotional'].append(step['emotional_attention'])
        per_emotion[e]['prompt'].append(step['prompt_attention'])
        per_emotion[e]['generated'].append(step['generated_attention'])
        per_emotion[e]['emotional'].append(step['emotional_attention'])

        lw_prompt += np.array(step['layerwise']['prompt_attention'])
        lw_gen    += np.array(step['layerwise']['generated_attention'])
        lw_emo    += np.array(step['layerwise']['emotional_attention'])

        # Head-level: (layers, heads)
        head_emo_arr = np.array(step['layerwise']['head_emo_attention'])  # (layers, heads)
        lw_head_emo  += head_emo_arr
        lw_head_var  += np.array(step['layerwise']['head_emo_variance'])  # (layers,)

        # Mean across layers for this token → (heads,)
        head_emo_global += np.array(step['mean_head_emo'])
        lw_count += 1

# Normalise
lw_prompt_mean   = (lw_prompt       / max(lw_count, 1)).tolist()
lw_gen_mean      = (lw_gen          / max(lw_count, 1)).tolist()
lw_emo_mean      = (lw_emo          / max(lw_count, 1)).tolist()
lw_head_emo_mean = (lw_head_emo     / max(lw_count, 1)).tolist()   # (layers, heads)
lw_head_var_mean = (lw_head_var     / max(lw_count, 1)).tolist()   # (layers,)
head_emo_global  = (head_emo_global / max(lw_count, 1)).tolist()   # (heads,)

# Which head is most emotionally attentive on average across all layers?
top_head_overall = int(np.argmax(head_emo_global))

summary = {
    'input_file':     INPUT_FILE,
    'n_samples':      len(per_sample),
    'n_tokens':       sum(r['gen_len'] for r in per_sample),
    'n_total_steps':  lw_count,
    'overall': {
        'mean_prompt_attention':        _mean(overall['prompt']),
        'mean_generated_attention':     _mean(overall['generated']),
        'mean_emotional_attention':     _mean(overall['emotional']),
        'last_10_prompt_attention':     _last_n_mean(overall['prompt']),
        'last_10_generated_attention':  _last_n_mean(overall['generated']),
    },
    'layerwise_mean': {
        'prompt_attention':    [round(x, 4) for x in lw_prompt_mean],
        'generated_attention': [round(x, 4) for x in lw_gen_mean],
        'emotional_attention': [round(x, 4) for x in lw_emo_mean],
        # per-layer variance across heads (how much heads disagree on emotional tokens)
        'head_emo_variance':   [round(x, 6) for x in lw_head_var_mean],
        # per-layer per-head mean emotional attention: list[layer][head]
        'head_emo_attention':  [[round(v, 4) for v in row] for row in lw_head_emo_mean],
    },
    # Global per-head mean emotional attention (averaged across layers and tokens)
    'head_summary': {
        'mean_emo_per_head':  [round(v, 4) for v in head_emo_global],
        'top_emo_head':       top_head_overall,
        'top_5_emo_heads':    [int(i) for i in np.argsort(head_emo_global)[-5:][::-1]],
        'bottom_5_emo_heads': [int(i) for i in np.argsort(head_emo_global)[:5]],
    },
    'per_emotion': {
        e: {
            'mean_prompt_attention':    _mean(v['prompt']),
            'mean_generated_attention': _mean(v['generated']),
            'mean_emotional_attention': _mean(v['emotional']),
        }
        for e, v in per_emotion.items()
    },
}

# %%
# ── Save ──────────────────────────────────────────────────────────────────────
with open(OUTPUT_FILE, 'w') as f:
    json.dump({'summary': summary, 'per_sample': per_sample}, f, indent=2)

print(f"\nSaved: {OUTPUT_FILE}")
print(json.dumps(summary, indent=2))
