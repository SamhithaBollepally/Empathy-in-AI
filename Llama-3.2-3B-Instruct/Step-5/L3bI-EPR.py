# %%
# ============================================================================
# Step 5 — Emotional Prototype Redistribution (EPR)
#
# Research Questions:
#   Q1. Can the model's internal emotional representation guide generation
#       without explicitly specifying the target emotion?
#   Q2. What happens when that representation is redistributed across layers?
#
# Pipeline (two phases):
#
# Phase 1 — Pre-computation (CPU, runs before model is loaded):
#   1. Source vector : context_embeddings[e][SOURCE_LAYER]
#                      (mean of 200 Set-1 prompt embeddings per emotion at layer 15)
#   2. Denoise       : PCA(50 components) fit on all 2200 individual prompt
#                      embeddings at SOURCE_LAYER → project + reconstruct
#   3. Weight        : L2 probe at SOURCE_LAYER → tanh → 11 signed weights
#                      (dominant emotion > 0, suppressed emotions < 0)
#   4. Inject vector : tanh_weights @ proto_matrix[L] for every target layer L,
#                      then unit-normalised → stored as float16
#
# Phase 2 — Generation (GPU):
#   - Hooks attach once at all STEER_LAYERS; active emotion switches per batch
#   - Norm-matched injection: hidden += coeff × ‖hidden‖ × inject_unit_vec
#   - Zero-shot prompting (same system prompt as Step-3 baseline)
#   - Unsteered judge scores each response (hooks disabled during judging)
#   - Checkpoint: saves after each emotion, skips completed emotions on resume
# ============================================================================

# %%
import os
import gc
import json
import pickle
import getpass
import numpy as np
import torch
import torch.nn as nn
import huggingface_hub
from collections import defaultdict
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForCausalLM

# %%
# ── Config ───────────────────────────────────────────────────────────────────
BASE_PATH  = '/Volumes/Reading/Projects/Empathy-LLM/Llama-3.2-3B-Instruct/'
# BASE_PATH = '/content/drive/MyDrive/Project-4/'

STEP1_PATH = BASE_PATH + 'Step-1/'
STEP2_PATH = BASE_PATH + 'Step-2/'
STEP5_PATH = BASE_PATH + 'Step-5/'
os.makedirs(STEP5_PATH, exist_ok=True)

EMOTIONS = ['afraid', 'angry', 'anxious', 'devastated', 'lonely', 'sad',
            'terrified', 'furious', 'grateful', 'hopeful', 'faithful']

SOURCE_LAYER   = 15                  # best L2 probe layer — emotional source
STEER_LAYERS   = list(range(1, 29)) # redistribute across all 28 decoder layers
PCA_COMPONENTS = 50                  # denoising components
COEFF          = 0.3                 # norm-matched steering coefficient
BATCH_SIZE     = 16                  # tuned for A100
GEN_MAX_NEW_TOKENS  = 100
EVAL_MAX_NEW_TOKENS = 5

# %%
# ── Phase 1a: Load context embeddings ────────────────────────────────────────
# context_embeddings.npz keys: "{emotion}_{layer}"
raw = np.load(STEP1_PATH + 'context_embeddings.npz')
context_embs = defaultdict(dict)
for key in raw.files:
    emotion, layer_idx = key.rsplit('_', 1)
    context_embs[emotion][int(layer_idx)] = raw[key].astype(np.float32)
del raw
print(f"Context embeddings loaded: {len(context_embs)} emotions × {len(next(iter(context_embs.values())))} layers")

# %%
# ── Phase 1b: Collect prompt embeddings at SOURCE_LAYER for PCA ──────────────
# prompt_embeddings.npz keys: "{emotion}_{prompt_idx}_{layer}"
# Only load SOURCE_LAYER to avoid pulling the full ~2200×29×3072 array into RAM.
print(f"Loading prompt embeddings at layer {SOURCE_LAYER} for PCA fitting...")
raw      = np.load(STEP1_PATH + 'prompt_embeddings.npz')
src_rows = [raw[key] for key in raw.files if key.rsplit('_', 2)[2] == str(SOURCE_LAYER)]
X_source = np.stack(src_rows, axis=0).astype(np.float32)   # (2200, 3072)
del raw, src_rows
gc.collect()
print(f"  Shape: {X_source.shape}")

# %%
# ── Phase 1c: Fit PCA ────────────────────────────────────────────────────────
pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
pca.fit(X_source)
explained = pca.explained_variance_ratio_.sum()
print(f"PCA fitted — {PCA_COMPONENTS} components explain {explained:.1%} of variance")

with open(STEP5_PATH + f'pca_l{SOURCE_LAYER}_k{PCA_COMPONENTS}.pkl', 'wb') as f:
    pickle.dump(pca, f)
print("PCA model saved.")

del X_source
gc.collect()

# %%
# ── Phase 1d: Load L2 probe at SOURCE_LAYER ──────────────────────────────────
bundle = torch.load(STEP2_PATH + 'emotion_probes.pt', map_location='cpu', weights_only=False)

# Probe class order is sorted alphabetically — must match proto_stack row order.
classes = [str(c) for c in bundle['classes']]   # ['afraid','angry','anxious',...]
scaler  = bundle['layers'][SOURCE_LAYER]['scaler']

probe = nn.Linear(bundle['hidden_dim'], len(classes), bias=True)
probe.load_state_dict(bundle['layers'][SOURCE_LAYER]['l2_state_dict'])
probe.eval()

W = probe.weight.detach().numpy()   # (11, 3072)
b = probe.bias.detach().numpy()     # (11,)
print(f"L2 probe loaded — layer {SOURCE_LAYER} | classes: {classes}")

# %%
# ── Phase 1e: Compute injection vectors ──────────────────────────────────────
# proto_stack[L]: (11, 3072) — all emotion prototypes at layer L, in probe class order
proto_stack = {
    L: np.stack([context_embs[e][L] for e in classes], axis=0).astype(np.float32)
    for L in STEER_LAYERS
}

injection_vecs = {}   # {emotion: {layer: np.ndarray (3072,) float16}}

print("\nComputing injection vectors:")
for e in EMOTIONS:
    # 1. Source: prototype at SOURCE_LAYER (raw, unscaled)
    h_proto = context_embs[e][SOURCE_LAYER]                              # (3072,)

    # 2. Denoise: project onto 50-dim emotional subspace, reconstruct
    h_denoised = pca.inverse_transform(
        pca.transform(h_proto.reshape(1, -1))
    ).squeeze().astype(np.float32)                                        # (3072,)

    # 3. Scale → probe scores → tanh weights
    h_scaled = scaler.transform(h_denoised.reshape(1, -1)).squeeze()    # (3072,)
    scores   = W @ h_scaled + b                                          # (11,)
    weights  = np.tanh(scores).astype(np.float32)                       # (11,) ∈ (−1, +1)

    # 4. Per-layer injection vector: weighted sum of prototypes → unit-normalise
    injection_vecs[e] = {}
    for L in STEER_LAYERS:
        h_inject = weights @ proto_stack[L]                              # (3072,)
        norm     = np.linalg.norm(h_inject)
        injection_vecs[e][L] = (
            (h_inject / norm) if norm > 1e-8 else h_inject
        ).astype(np.float16)

    top3 = sorted(zip(weights.tolist(), classes), reverse=True)[:3]
    print(f"  {e:12s}: top weights → {[(round(w, 3), c) for w, c in top3]}")

# Save (compressed) for reproducibility
np.savez_compressed(
    STEP5_PATH + 'injection_vectors.npz',
    **{f"{e}_{L}": injection_vecs[e][L] for e in EMOTIONS for L in STEER_LAYERS}
)
print(f"\nInjection vectors saved → injection_vectors.npz  "
      f"({len(EMOTIONS)} emotions × {len(STEER_LAYERS)} layers)")

del proto_stack
gc.collect()

# %%
# ── Phase 2a: Load model ─────────────────────────────────────────────────────
token = getpass.getpass("HuggingFace token: ")
huggingface_hub.login(token=token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU : {props.name} | VRAM: {props.total_memory / 1e9:.1f} GB")

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = 'left'   # left-pad for batched decoder-only generation

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
)
model.eval()
MODEL_DEVICE = next(model.parameters()).device
NUM_LAYERS   = model.config.num_hidden_layers
print(f"Model loaded — {NUM_LAYERS} decoder layers | device: {MODEL_DEVICE}")

# %%
# ── Phase 2b: Redistribution hook ────────────────────────────────────────────

class EmotionRedistributor:
    """
    Attaches one persistent forward hook per target layer.

    Design:
    - Hooks are registered once and kept alive for the full generation loop.
    - Switching the active emotion is O(1) (attribute write, no hook re-attachment).
    - Injection vectors are pre-loaded as fp32 tensors on the model device.
    - Injection formula (norm-matched):
          hidden += coeff × ‖hidden‖ × unit_vec
      so the relative nudge is consistent across layers regardless of their scale.
    """

    def __init__(self, model, injection_vecs, coeff, steer_layers, device):
        self.model          = model
        self.coeff          = coeff
        self.steer_layers   = steer_layers
        self.active_emotion = None
        self.handles        = []

        # Pre-move all vectors to device as fp32 (avoids repeated host→device copies
        # and keeps hook arithmetic numerically stable under fp16 hidden states).
        self._vecs = {
            e: {
                L: torch.tensor(injection_vecs[e][L], dtype=torch.float32, device=device)
                for L in steer_layers
            }
            for e in injection_vecs
        }

    def _make_hook(self, layer_idx):
        def hook(module, inputs, output):
            if self.active_emotion is None:
                return output
            vec      = self._vecs[self.active_emotion][layer_idx]   # (3072,)
            is_tuple = isinstance(output, tuple)
            hidden   = output[0] if is_tuple else output            # (batch, seq, 3072)
            norm     = hidden.detach().float().norm(dim=-1, keepdim=True)
            push     = vec.to(hidden.dtype)
            hidden   = hidden + (self.coeff * norm).to(hidden.dtype) * push
            return (hidden,) + tuple(output[1:]) if is_tuple else hidden
        return hook

    def attach(self):
        for L in self.steer_layers:
            handle = self.model.model.layers[L - 1].register_forward_hook(
                self._make_hook(L)
            )
            self.handles.append(handle)
        print(f"Hooks attached at {len(self.steer_layers)} layers.")

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def set_emotion(self, emotion):
        self.active_emotion = emotion

    def disable(self):
        self.active_emotion = None

# %%
# ── Phase 2c: Generation helpers ─────────────────────────────────────────────

GEN_SYSTEM_PROMPT = (
    "You are a medical assistant. Given the user's emotional situation, do two things:\n"
    "1. Write a response (1-2 sentences).\n"
    "2. Rate how emotional is the user's situation on a scale of 1 (mild) to 5 (highly emotional).\n\n"
    "Format your output exactly as:\n"
    "Response: <your response>\n"
    "Intensity: <1-5>"
)

def build_eval_prompt(response):
    return (
        "Rate how emotionally expressive this response is on a scale of 1 (mild) to 5 (highly emotional). "
        "Reply with only a single number.\n\nResponse: " + response
    )

def format_chat(system, user):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user",   "content": user})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def generate_batch(prompts, max_new_tokens, do_sample=True):
    """Batched generation over a list of pre-formatted chat strings."""
    outputs = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch  = prompts[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch, return_tensors='pt', padding=True,
            truncation=True, max_length=512
        ).to(MODEL_DEVICE)
        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=0.7 if do_sample else None,
                top_p=0.9     if do_sample else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs['input_ids'].shape[1]
        for ids in out_ids:
            outputs.append(
                tokenizer.decode(ids[input_len:], skip_special_tokens=True).strip()
            )
        del inputs, out_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return outputs

def parse_score(text):
    """Return first digit 1-5 found in text, else None."""
    for ch in text:
        if ch in '12345':
            return int(ch)
    return None

def parse_response_intensity(text):
    """Split 'Response: ...\nIntensity: X' → (response_str, intensity_int).
    Falls back to raw text as response and None intensity if format is not followed.
    """
    response, intensity = text, None
    for line in text.split('\n'):
        s = line.strip()
        if s.lower().startswith('response:'):
            response  = s[len('response:'):].strip()
        elif s.lower().startswith('intensity:'):
            intensity = parse_score(s[len('intensity:'):])
    return response, intensity

# %%
# ── Phase 2d: Generate with redistribution ───────────────────────────────────

with open(BASE_PATH + 'prompts_utterances_set2.json') as f:
    data = json.load(f)

out_path = STEP5_PATH + f'epr_photocopy_c{COEFF}_pca{PCA_COMPONENTS}.json'
log_path = STEP5_PATH + 'epr_log.json'

# Resume support: load checkpoint if the run was interrupted
if os.path.exists(out_path):
    with open(out_path) as f:
        results = json.load(f)
    print(f"Resuming — already done: {list(results.keys())}")
else:
    results = {}

redistributor = EmotionRedistributor(
    model, injection_vecs, COEFF, STEER_LAYERS, MODEL_DEVICE
)
redistributor.attach()

for emotion in EMOTIONS:
    if emotion in results:
        print(f"Skipping {emotion} (checkpoint found).")
        continue

    print(f"\n── {emotion} ({len(data[emotion])} utterances) ──")
    redistributor.set_emotion(emotion)

    items      = data[emotion]
    utterances = [it['first_utterance'] for it in items]
    prompts_   = [it['prompt']          for it in items]

    # Stage 1: steered generation
    chats  = [format_chat(GEN_SYSTEM_PROMPT, u) for u in utterances]
    raw    = generate_batch(chats, GEN_MAX_NEW_TOKENS, do_sample=True)
    parsed = [parse_response_intensity(r) for r in raw]

    # Stage 2: judge without steering (hooks disabled, reads text neutrally)
    redistributor.disable()
    eval_chats = [format_chat(None, build_eval_prompt(resp)) for resp, _ in parsed]
    judge_out  = generate_batch(eval_chats, EVAL_MAX_NEW_TOKENS, do_sample=False)

    results[emotion] = [
        {
            "emotion":        emotion,
            "prompt":         p,
            "utterance":      u,
            "response":       resp,
            "self_intensity": inten,
            "judged_score":   parse_score(jo),
            "coeff":          COEFF,
            "source_layer":   SOURCE_LAYER,
            "pca_components": PCA_COMPONENTS,
            "method":         "epr_photocopy",
        }
        for p, u, (resp, inten), jo
        in zip(prompts_, utterances, parsed, judge_out)
    ]

    # Checkpoint after each emotion
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    judged = [it['judged_score']   for it in results[emotion] if it['judged_score']   is not None]
    intens = [it['self_intensity'] for it in results[emotion] if it['self_intensity'] is not None]
    print(f"  judged_mean={round(float(np.mean(judged)), 4) if judged else None}  "
          f"intensity_mean={round(float(np.mean(intens)), 4) if intens else None}  "
          f"| checkpoint saved ({len(results)}/{len(EMOTIONS)} emotions)")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

redistributor.detach()

# %%
# ── Save summary log ─────────────────────────────────────────────────────────

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
    'method':              'epr_photocopy',
    'coeff':               COEFF,
    'source_layer':        SOURCE_LAYER,
    'steer_layers':        STEER_LAYERS,
    'pca_components':      PCA_COMPONENTS,
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
