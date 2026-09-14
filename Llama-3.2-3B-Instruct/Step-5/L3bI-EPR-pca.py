# %%
# ============================================================================
# Step 5c — Emotional Prototype Redistribution with PCA Distribution
#
# Difference from L3bI-EPR.py (photocopy):
#   Photocopy: same tanh-weighted prototype blend injected at every layer.
#   PCA:       decompose the source emotion vector into principal components
#              at layer 15, then reconstruct it per target layer using THAT
#              layer's own emotion basis. Early layers receive fewer components
#              (broad emotional tone); late layers receive more (specific emotion).
#
# Pipeline:
#   1. Source h_emotion = tanh(probe(h_denoised)) @ proto_stack[15]
#   2. Project h_emotion onto PCA axes fit on the 11 prototypes at layer 15
#      → component vector (n_comp,)
#   3. Fit PCA on the 11 prototypes at every target layer L
#   4. For each layer L, mask the component vector (layer-dependent) and
#      reconstruct h_inject_L in layer L's PCA basis
#   5. Unit-normalise and inject with norm-matched scaling
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

SOURCE_LAYER      = 15
STEER_LAYERS      = list(range(1, 29))
PCA_SOURCE_COMP   = 11    # components for source decomposition (max = 11 prototypes)
PCA_DENOISE_COMP  = 50    # K for initial denoising PCA on individual prompts
COEFF             = 0.2
BATCH_SIZE        = 16    # reduce slightly because hooks run on all 28 layers
GEN_MAX_NEW_TOKENS  = 100
EVAL_MAX_NEW_TOKENS = 5

# How to distribute components across layers:
#   'ramp'  → linearly increase active components with depth
#   'step'  → early=1, mid=half, late=all
#   'full'  → every layer gets all components (PCA basis adaptation only)
PCA_DIST_MODE = 'ramp'

# %%
def component_mask(layer_idx, n_comp, mode):
    """Return binary mask of length n_comp for given layer."""
    if mode == 'ramp':
        max_c = max(1, min(n_comp, int(np.ceil(layer_idx / 28 * n_comp))))
    elif mode == 'step':
        if layer_idx <= 10:
            max_c = 1
        elif layer_idx <= 20:
            max_c = n_comp // 2
        else:
            max_c = n_comp
    elif mode == 'full':
        max_c = n_comp
    else:
        raise ValueError(f"Unknown PCA_DIST_MODE: {mode}")
    mask = np.zeros(n_comp)
    mask[:max_c] = 1.0
    return mask

# %%
# ── Phase 1a: Load context embeddings ────────────────────────────────────────
raw = np.load(STEP1_PATH + 'context_embeddings.npz')
context_embs = defaultdict(dict)
for key in raw.files:
    emotion, layer_idx = key.rsplit('_', 1)
    context_embs[emotion][int(layer_idx)] = raw[key].astype(np.float32)
del raw
gc.collect()
print(f"Loaded context embeddings: {len(context_embs)} emotions × layers")

# %%
# ── Phase 1b: Prompt embeddings for denoising PCA ────────────────────────────
print(f"Loading prompt embeddings at layer {SOURCE_LAYER} for denoising PCA...")
raw = np.load(STEP1_PATH + 'prompt_embeddings.npz')
src_rows = [raw[key] for key in raw.files if key.rsplit('_', 2)[2] == str(SOURCE_LAYER)]
X_source = np.stack(src_rows, axis=0).astype(np.float32)
del raw, src_rows
gc.collect()
print(f"  shape: {X_source.shape}")

# %%
# ── Phase 1c: Fit denoising PCA ──────────────────────────────────────────────
pca_denoise = PCA(n_components=PCA_DENOISE_COMP, random_state=42)
pca_denoise.fit(X_source)
print(f"Denoising PCA explains {pca_denoise.explained_variance_ratio_.sum():.1%} of variance")

with open(STEP5_PATH + f'pca_denoise_l{SOURCE_LAYER}_k{PCA_DENOISE_COMP}.pkl', 'wb') as f:
    pickle.dump(pca_denoise, f)

del X_source
gc.collect()

# %%
# ── Phase 1d: Load L2 probe at source layer ──────────────────────────────────
bundle = torch.load(STEP2_PATH + 'emotion_probes.pt', map_location='cpu', weights_only=False)
classes = [str(c) for c in bundle['classes']]
scaler  = bundle['layers'][SOURCE_LAYER]['scaler']

probe = nn.Linear(bundle['hidden_dim'], len(classes), bias=True)
probe.load_state_dict(bundle['layers'][SOURCE_LAYER]['l2_state_dict'])
probe.eval()
W = probe.weight.detach().numpy()
b = probe.bias.detach().numpy()
print(f"Probe loaded — classes: {classes}")

# %%
# ── Phase 1e: Build per-layer prototype stacks ───────────────────────────────
proto_stack = {
    L: np.stack([context_embs[e][L] for e in classes], axis=0).astype(np.float32)
    for L in [SOURCE_LAYER] + STEER_LAYERS
}

# %%
# ── Phase 1f: Fit per-layer PCA on prototypes ────────────────────────────────
# pcas[L].components_ gives the emotion axes at layer L (in the order of classes)
pcas = {}
for L in [SOURCE_LAYER] + STEER_LAYERS:
    n = min(PCA_SOURCE_COMP, proto_stack[L].shape[0])
    pca = PCA(n_components=n, random_state=42)
    pca.fit(proto_stack[L])
    pcas[L] = pca
    exp = pca.explained_variance_ratio_.sum()
    if L == SOURCE_LAYER:
        print(f"Source PCA (layer {L}) {n} comps explains {exp:.1%} variance")

# %%
# ── Phase 1g: Compute PCA-distributed injection vectors ──────────────────────
injection_vecs = {}
print("\nComputing PCA-distributed injection vectors:")

for e in EMOTIONS:
    # 1. Denoise source prototype
    h_proto = context_embs[e][SOURCE_LAYER]
    h_denoised = pca_denoise.inverse_transform(
        pca_denoise.transform(h_proto.reshape(1, -1))
    ).squeeze().astype(np.float32)

    # 2. Probe → tanh weights
    h_scaled = scaler.transform(h_denoised.reshape(1, -1)).squeeze()
    scores   = W @ h_scaled + b
    weights  = np.tanh(scores).astype(np.float32)

    # 3. Source emotion vector in layer 15 space
    h_emotion = weights @ proto_stack[SOURCE_LAYER]   # (3072,)

    # 4. Project onto source-layer emotion axes
    src_comp = pcas[SOURCE_LAYER].transform(h_emotion.reshape(1, -1)).squeeze()  # (n_comp,)

    # 5. Reconstruct per target layer with masked components in layer's own basis
    injection_vecs[e] = {}
    for L in STEER_LAYERS:
        n_comp = pcas[L].n_components_
        mask   = component_mask(L, n_comp, PCA_DIST_MODE)
        masked = src_comp[:n_comp].copy()
        masked *= mask
        h_inject_L = pcas[L].inverse_transform(masked.reshape(1, -1)).squeeze()
        norm = np.linalg.norm(h_inject_L)
        injection_vecs[e][L] = (
            (h_inject_L / norm) if norm > 1e-8 else h_inject_L
        ).astype(np.float16)

    top3 = sorted(zip(weights.tolist(), classes), reverse=True)[:3]
    print(f"  {e:12s}: {[(round(w, 3), c) for w, c in top3]}")

np.savez_compressed(
    STEP5_PATH + f'injection_vectors_pca_{PCA_DIST_MODE}.npz',
    **{f"{e}_{L}": injection_vecs[e][L] for e in EMOTIONS for L in STEER_LAYERS}
)
print(f"Saved injection vectors → injection_vectors_pca_{PCA_DIST_MODE}.npz")

del proto_stack, pcas

# %%
# ── Phase 2a: Load model ─────────────────────────────────────────────────────
token = getpass.getpass("HuggingFace token: ")
huggingface_hub.login(token=token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | VRAM: {props.total_memory / 1e9:.1f} GB")

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = 'left'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
)
model.eval()
MODEL_DEVICE = next(model.parameters()).device
print(f"Model loaded on {MODEL_DEVICE}")

# %%
# ── Phase 2b: Redistribution hook ────────────────────────────────────────────
class EmotionRedistributor:
    def __init__(self, model, injection_vecs, coeff, steer_layers, device):
        self.model = model
        self.coeff = coeff
        self.steer_layers = steer_layers
        self.active_emotion = None
        self.handles = []
        self._vecs = {
            e: {L: torch.tensor(injection_vecs[e][L], dtype=torch.float32, device=device)
                for L in steer_layers}
            for e in injection_vecs
        }

    def _make_hook(self, layer_idx):
        def hook(module, inputs, output):
            if self.active_emotion is None:
                return output
            vec = self._vecs[self.active_emotion][layer_idx]
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            norm = hidden.detach().float().norm(dim=-1, keepdim=True)
            push = vec.to(hidden.dtype)
            hidden = hidden + (self.coeff * norm).to(hidden.dtype) * push
            return (hidden,) + tuple(output[1:]) if is_tuple else hidden
        return hook

    def attach(self):
        for L in self.steer_layers:
            h = self.model.model.layers[L - 1].register_forward_hook(self._make_hook(L))
            self.handles.append(h)
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
    messages.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def generate_batch(prompts, max_new_tokens, do_sample=True):
    outputs = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i + BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                           truncation=True, max_length=512).to(MODEL_DEVICE)
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
            outputs.append(tokenizer.decode(ids[input_len:], skip_special_tokens=True).strip())
        del inputs, out_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return outputs

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

# %%
# ── Phase 2d: Generate with PCA-distributed steering ─────────────────────────
with open(BASE_PATH + 'prompts_utterances_set2.json') as f:
    data = json.load(f)

subfolder = STEP5_PATH + f'Steering_distribution_pca_{PCA_DIST_MODE}/'
os.makedirs(subfolder, exist_ok=True)
out_path  = subfolder + f'epr_pca_{PCA_DIST_MODE}_c{COEFF}_pca{PCA_DENOISE_COMP}.json'
log_path  = subfolder + 'epr_pca_log.json'

if os.path.exists(out_path):
    with open(out_path) as f:
        results = json.load(f)
    print(f"Resuming from {list(results.keys())}")
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

    chats  = [format_chat(GEN_SYSTEM_PROMPT, u) for u in utterances]
    raw    = generate_batch(chats, GEN_MAX_NEW_TOKENS, do_sample=True)
    parsed = [parse_response_intensity(r) for r in raw]

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
            "pca_denoise":    PCA_DENOISE_COMP,
            "pca_dist_mode":  PCA_DIST_MODE,
            "method":         "epr_pca_distribution",
        }
        for p, u, (resp, inten), jo
        in zip(prompts_, utterances, parsed, judge_out)
    ]

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    judged = [it['judged_score']   for it in results[emotion] if it['judged_score']   is not None]
    intens = [it['self_intensity'] for it in results[emotion] if it['self_intensity'] is not None]
    print(f"  judged_mean={round(float(np.mean(judged)), 4) if judged else None}  "
          f"intensity_mean={round(float(np.mean(intens)), 4) if intens else None}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

redistributor.detach()

# %%
# ── Save summary log ─────────────────────────────────────────────────────────
per_emotion = []
for e in EMOTIONS:
    items = results.get(e, [])
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
    'method':              'epr_pca_distribution',
    'coeff':               COEFF,
    'source_layer':        SOURCE_LAYER,
    'steer_layers':        STEER_LAYERS,
    'pca_denoise':         PCA_DENOISE_COMP,
    'pca_dist_mode':       PCA_DIST_MODE,
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
