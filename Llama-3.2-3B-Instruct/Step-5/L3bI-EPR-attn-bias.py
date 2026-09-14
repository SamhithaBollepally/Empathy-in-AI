# %%
# ============================================================================
# Step 5k — Attention Bias Injection
#
# Adds a bias to the raw attention score matrix (before softmax) at target
# layers. Uses a forward PRE-hook to modify the attention_mask passed to
# self_attn, which Llama's eager attention adds directly to the Q@K^T scores:
#
#   attn_scores = Q @ K^T / sqrt(d) + attention_mask
#   attn_weights = softmax(attn_scores)
#
# By adding our bias to attention_mask, we shift probability mass before
# softmax without touching attn_output after the fact.
#
# Bias modes:
#   'prompt_boost'   → generated tokens attend more to ALL prompt tokens
#   'emotion_boost'  → generated tokens attend more to emotionally relevant
#                      prompt tokens (detected via L15 probe)
#   'self_suppress'  → generated tokens attend less to their own prior tokens
# ============================================================================

# %%
import os
import gc
import json
import getpass
import numpy as np
import torch
import torch.nn as nn
import huggingface_hub
from transformers import AutoTokenizer, AutoModelForCausalLM

# %%
# ── Config ───────────────────────────────────────────────────────────────────
BASE_PATH = '/Volumes/Reading/Projects/Empathy-LLM/Llama-3.2-3B-Instruct/'
# BASE_PATH = '/content/drive/MyDrive/Project-4/'

STEP2_PATH = BASE_PATH + 'Step-2/'
STEP5_PATH = BASE_PATH + 'Step-5/'
os.makedirs(STEP5_PATH, exist_ok=True)

EMOTIONS = ['afraid', 'angry', 'anxious', 'devastated', 'lonely', 'sad',
            'terrified', 'furious', 'grateful', 'hopeful', 'faithful']

BIAS_LAYERS = [20, 21, 22, 23, 24, 25, 26, 27, 28]
BIAS_MODE   = 'emotion_boost'   # 'prompt_boost' | 'emotion_boost' | 'self_suppress'
BIAS_VALUE  = 2.0               # added to log-space scores; ~7× multiplier per token
PROBE_LAYER = 15
N_ITEMS     = 50                # utterances per emotion (None = all)
BATCH_SIZE  = 16
GEN_MAX_NEW_TOKENS  = 60
EVAL_MAX_NEW_TOKENS = 5

# %%
# ── Load probe ────────────────────────────────────────────────────────────────
bundle = torch.load(STEP2_PATH + 'emotion_probes.pt', map_location='cpu', weights_only=False)
EMOTION_CLASSES = [str(c) for c in bundle['classes']]
CLASS_TO_IDX    = {c: i for i, c in enumerate(EMOTION_CLASSES)}

probe_scaler = bundle['layers'][PROBE_LAYER]['scaler']
probe_lin = nn.Linear(bundle['hidden_dim'], len(EMOTION_CLASSES), bias=True)
probe_lin.load_state_dict(bundle['layers'][PROBE_LAYER]['l2_state_dict'])
probe_lin.eval()

probe_W     = probe_lin.weight.detach().to(torch.float32)
probe_b     = probe_lin.bias.detach().to(torch.float32)
probe_mean  = torch.tensor(probe_scaler.mean_,  dtype=torch.float32)
probe_scale = torch.tensor(probe_scaler.scale_, dtype=torch.float32)
print(f"Probe loaded for layer {PROBE_LAYER}")

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
    attn_implementation="eager",   # required: SDPA does not expose attention_mask
)
model.eval()
MODEL_DEVICE = next(model.parameters()).device
print(f"Model loaded on {MODEL_DEVICE}")

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

def build_eval_prompt(response):
    return (
        "Rate how emotionally expressive this response is on a scale of 1 (mild) to 5 (highly emotional). "
        "Reply with only a single number.\n\nResponse: " + response
    )

def parse_score(text):
    for ch in text:
        if ch in '12345':
            return int(ch)
    return None

def parse_response_intensity(text):
    response, intensity = text, None
    for line in text.split('\n'):
        s = line.strip()
        if s.lower().startswith('response:'):
            response = s[len('response:'):].strip()
        elif s.lower().startswith('intensity:'):
            intensity = parse_score(s[len('intensity:'):])
    return response, intensity

def get_emo_positions(utterance, emotion):
    """
    Run one forward pass with output_hidden_states to get h15,
    then use the probe to find emotionally relevant prompt token positions.
    """
    target_idx = CLASS_TO_IDX.get(emotion)
    chat   = format_chat(GEN_SYSTEM_PROMPT, utterance)
    inputs = tokenizer(chat, return_tensors='pt', truncation=True,
                       max_length=512).to(MODEL_DEVICE)
    prompt_len = inputs['input_ids'].shape[1]
    with torch.inference_mode():
        out = model(**inputs, output_hidden_states=True)
    h15 = out.hidden_states[PROBE_LAYER][0, :prompt_len, :].float()
    scaled = (h15 - probe_mean.to(h15.device)) / probe_scale.to(h15.device)
    scores = scaled @ probe_W.to(h15.device).T + probe_b.to(h15.device)
    pred   = scores.argmax(dim=-1).cpu().tolist()
    emo_pos = [i for i, p in enumerate(pred) if p == target_idx] if target_idx is not None else []
    del inputs, out, h15
    return prompt_len, emo_pos

# %%
# ── Attention bias hook (pre-hook on attention_mask) ─────────────────────────
class AttnBiasState:
    """
    Shared mutable state read by all pre-hooks.
    Set before each generate call.
    """
    def __init__(self):
        self.active        = False
        self.prompt_len    = 0     # token count of the un-padded prompt in the batch
        self.emo_positions = []    # emotionally relevant positions in the prompt

state = AttnBiasState()

def make_pre_hook(layer_idx):
    """
    Returns a forward pre-hook for self_attn at layer_idx.
    Modifies the attention_mask in kwargs to add the bias BEFORE softmax.

    During cached generation each forward call processes one new query token.
    attn_mask shape: (batch, 1, q_len, k_len)
      - q_len = 1 for cached steps, prompt_len for prefill
      - k_len = total sequence length seen so far

    We only add the bias on cached steps (q_len == 1) because the query is
    a generated token. During prefill all queries are prompt tokens.
    """
    def pre_hook(module, args, kwargs):
        if not state.active or state.prompt_len == 0:
            return args, kwargs

        attn_mask = kwargs.get('attention_mask')
        if attn_mask is None:
            return args, kwargs

        _, _, q_len, k_len = attn_mask.shape

        # Only apply during cached single-token generation steps
        if q_len != 1:
            return args, kwargs

        bias = torch.zeros_like(attn_mask, dtype=torch.float32)
        p    = state.prompt_len

        if BIAS_MODE == 'prompt_boost':
            # Boost all prompt key positions
            p_end = min(p, k_len)
            bias[:, :, :, :p_end] += BIAS_VALUE

        elif BIAS_MODE == 'emotion_boost':
            # Boost only probe-identified emotional prompt positions
            for pos in state.emo_positions:
                if pos < k_len:
                    bias[:, :, :, pos] += BIAS_VALUE

        elif BIAS_MODE == 'self_suppress':
            # Suppress already-generated key positions
            if p < k_len:
                bias[:, :, :, p:] -= BIAS_VALUE

        kwargs['attention_mask'] = (attn_mask.float() + bias).to(attn_mask.dtype)
        return args, kwargs
    return pre_hook

# Attach pre-hooks to each target layer
hooks = []
for L in BIAS_LAYERS:
    h = model.model.layers[L - 1].self_attn.register_forward_pre_hook(
        make_pre_hook(L), with_kwargs=True
    )
    hooks.append(h)
print(f"Pre-hooks attached at layers {BIAS_LAYERS}, mode={BIAS_MODE}, value={BIAS_VALUE}")

# %%
# ── Load data ─────────────────────────────────────────────────────────────────
with open(BASE_PATH + 'prompts_utterances_set2.json') as f:
    data = json.load(f)

# %%
# ── Output paths ──────────────────────────────────────────────────────────────
layer_tag = ','.join(map(str, BIAS_LAYERS))
out_path = os.path.join(STEP5_PATH,
    f'epr_attnbias_{BIAS_MODE}_b{BIAS_VALUE}_l{layer_tag}.json')
log_path = os.path.join(STEP5_PATH,
    f'epr_attnbias_{BIAS_MODE}_b{BIAS_VALUE}_log.json')

if os.path.exists(out_path):
    with open(out_path) as f:
        results = json.load(f)
    print(f"Resuming from checkpoint: {list(results.keys())}")
else:
    results = {}

# %%
# ── Generate ──────────────────────────────────────────────────────────────────
for emotion in EMOTIONS:
    if emotion in results:
        print(f"Skipping {emotion} (checkpoint found).")
        continue

    items = data[emotion] if N_ITEMS is None else data[emotion][:N_ITEMS]
    print(f"\n── {emotion} ({len(items)} utterances) ──")
    results[emotion] = []

    for i in range(0, len(items), BATCH_SIZE):
        batch  = items[i:i + BATCH_SIZE]
        prompts_    = [it['prompt']          for it in batch]
        utterances  = [it['first_utterance'] for it in batch]

        # Get emotional positions for each sample; use the minimum prompt length
        # as the safe shared lower bound for the batched run
        batch_plens = []
        batch_epos  = []
        for u in utterances:
            plen, epos = get_emo_positions(u, emotion)
            batch_plens.append(plen)
            batch_epos.append(epos)

        min_plen = min(batch_plens)
        # Merge emotional positions that fall within the minimum prompt window
        merged_epos = sorted(set(p for ep in batch_epos for p in ep if p < min_plen))

        # Arm the state
        state.prompt_len    = min_plen
        state.emo_positions = merged_epos
        state.active        = True

        chats  = [format_chat(GEN_SYSTEM_PROMPT, u) for u in utterances]
        inputs = tokenizer(chats, return_tensors='pt', padding=True,
                           truncation=True, max_length=512).to(MODEL_DEVICE)
        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs['input_ids'].shape[1]
        raw    = [tokenizer.decode(ids[input_len:], skip_special_tokens=True).strip()
                  for ids in out_ids]
        parsed = [parse_response_intensity(r) for r in raw]

        # Judge without bias
        state.active = False
        eval_chats  = [format_chat(None, build_eval_prompt(resp)) for resp, _ in parsed]
        eval_inputs = tokenizer(eval_chats, return_tensors='pt', padding=True,
                                truncation=True, max_length=512).to(MODEL_DEVICE)
        with torch.inference_mode():
            judge_ids = model.generate(
                **eval_inputs,
                max_new_tokens=EVAL_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        j_len     = eval_inputs['input_ids'].shape[1]
        judge_out = [tokenizer.decode(ids[j_len:], skip_special_tokens=True).strip()
                     for ids in judge_ids]

        for p, u, (resp, inten), jo in zip(prompts_, utterances, parsed, judge_out):
            results[emotion].append({
                'emotion':        emotion,
                'prompt':         p,
                'utterance':      u,
                'response':       resp,
                'self_intensity': inten,
                'judged_score':   parse_score(jo),
                'bias_mode':      BIAS_MODE,
                'bias_value':     BIAS_VALUE,
                'bias_layers':    BIAS_LAYERS,
                'method':         'epr_attnbias',
            })

        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

        del inputs, out_ids, eval_inputs, judge_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    judged = [it['judged_score']   for it in results[emotion] if it['judged_score']   is not None]
    intens = [it['self_intensity'] for it in results[emotion] if it['self_intensity'] is not None]
    print(f"  judged={round(float(np.mean(judged)),4) if judged else None}  "
          f"intensity={round(float(np.mean(intens)),4) if intens else None}")

# Remove hooks when done
for h in hooks:
    h.remove()
hooks.clear()
print("Hooks removed.")

# %%
# ── Summary log ───────────────────────────────────────────────────────────────
per_emotion = []
for e in EMOTIONS:
    items  = results.get(e, [])
    judged = [it['judged_score']   for it in items if it.get('judged_score')   is not None]
    intens = [it['self_intensity'] for it in items if it.get('self_intensity') is not None]
    per_emotion.append({
        'emotion':             e,
        'n':                   len(items),
        'judged_score_mean':   round(float(np.mean(judged)), 4) if judged else None,
        'self_intensity_mean': round(float(np.mean(intens)), 4) if intens else None,
    })

all_judged = [r['judged_score_mean'] for r in per_emotion if r['judged_score_mean'] is not None]
log = {
    'method':              'epr_attnbias',
    'bias_mode':           BIAS_MODE,
    'bias_value':          BIAS_VALUE,
    'bias_layers':         BIAS_LAYERS,
    'output_file':         os.path.basename(out_path),
    'overall_judged_mean': round(float(np.mean(all_judged)), 4) if all_judged else None,
    'per_emotion':         per_emotion,
}
with open(log_path, 'w') as f:
    json.dump(log, f, indent=2)

# %%
print(f"\nResults : {out_path}")
print(f"Log     : {log_path}")
print(f"Overall judged mean: {log['overall_judged_mean']}")
