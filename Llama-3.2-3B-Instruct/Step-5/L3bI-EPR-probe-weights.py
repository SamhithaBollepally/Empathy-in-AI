# %%
# ============================================================================
# Step 5l — Probe Weight Analysis
#
# Analyses the L2 probe weight matrix W (11 × 3072) at every layer to answer:
#   1. How large is each emotion's weight vector? (norm per emotion per layer)
#   2. How much do weights vary across dimensions? (variance per layer)
#   3. Which dimensions matter most per emotion? (top-k dims per emotion)
#   4. How similar are emotion directions to each other? (cosine similarity)
#   5. How does weight structure change across layers?
#
# No model or GPU required — runs on saved emotion_probes.pt only.
# ============================================================================

# %%
import json
import numpy as np
import torch
from pathlib import Path

# %%
# ── Config ───────────────────────────────────────────────────────────────────
BASE_PATH  = Path('/Volumes/Reading/Projects/Empathy-LLM/Llama-3.2-3B-Instruct/')
# BASE_PATH = Path('/content/drive/MyDrive/Project-4/')

STEP2_PATH = BASE_PATH / 'Step-2'
STEP5_PATH = BASE_PATH / 'Step-5'
STEP5_PATH.mkdir(exist_ok=True)

TOP_K = 10   # top dimensions to report per emotion

# %%
# ── Load probe bundle ─────────────────────────────────────────────────────────
bundle = torch.load(STEP2_PATH / 'emotion_probes.pt', map_location='cpu', weights_only=False)
EMOTIONS   = [str(c) for c in bundle['classes']]   # 11 emotion labels
N_LAYERS   = len(bundle['layers'])                  # 0..28
HIDDEN_DIM = bundle['hidden_dim']                   # 3072
print(f"Emotions : {EMOTIONS}")
print(f"Layers   : {N_LAYERS}  (0 = embeddings, 1-28 = decoder)")
print(f"Hidden   : {HIDDEN_DIM}")

# %%
# ── Extract W per layer ───────────────────────────────────────────────────────
# W shape per layer: (11, 3072)
#   Row e = weight vector for emotion e
#   W[e, d] = how much dimension d contributes to predicting emotion e

layer_weights = {}
for L in range(N_LAYERS):
    if L not in bundle['layers']:
        continue
    lin = torch.nn.Linear(HIDDEN_DIM, len(EMOTIONS), bias=True)
    lin.load_state_dict(bundle['layers'][L]['l2_state_dict'])
    layer_weights[L] = lin.weight.detach().numpy()   # (11, 3072)

print(f"Loaded weights for {len(layer_weights)} layers.")

# %%
# ── 1. Emotion weight norms per layer ─────────────────────────────────────────
# ||w_e||  for each emotion e at each layer
# A larger norm means the probe relies more heavily on the hidden-state
# dimensions collectively to predict emotion e at that layer.

norms = {}   # norms[L] = (11,) array
for L, W in layer_weights.items():
    norms[L] = np.linalg.norm(W, axis=1)   # (11,)

# Print: for each layer, which emotion has the highest norm?
print("\n── Emotion weight norms per layer (top emotion) ──")
print(f"{'Layer':>6}  {'Top emotion':>12}  {'Norm':>8}  {'Min norm':>9}  {'Max norm':>9}  {'Variance':>10}")
for L in sorted(norms.keys()):
    n   = norms[L]
    top = EMOTIONS[np.argmax(n)]
    print(f"{L:>6}  {top:>12}  {np.max(n):>8.4f}  {np.min(n):>9.4f}  {np.max(n):>9.4f}  {n.var():>10.6f}")

# %%
# ── 2. Weight variance across dimensions per layer ────────────────────────────
# var(W_L) averaged across emotions.
# High variance = probe uses a diverse set of dimensions.
# Low variance  = weights are concentrated near zero (uniform uncertainty).

print("\n── Mean weight variance across all dimensions per layer ──")
print(f"{'Layer':>6}  {'Mean var (across dims & emotions)':>35}")
for L in sorted(layer_weights.keys()):
    W       = layer_weights[L]               # (11, 3072)
    var_per_emotion = W.var(axis=1)          # (11,)  variance over 3072 dims
    mean_var = var_per_emotion.mean()
    print(f"{L:>6}  {mean_var:>35.8f}")

# %%
# ── 3. Top-k dimensions per emotion at the best probe layer ──────────────────
# Probe layer 15 has the highest accuracy; examine which dimensions it uses.

BEST_LAYER = 15
W_best = layer_weights[BEST_LAYER]   # (11, 3072)

print(f"\n── Top-{TOP_K} dimensions per emotion at layer {BEST_LAYER} ──")
top_dims = {}
for e_idx, emotion in enumerate(EMOTIONS):
    w_e   = W_best[e_idx]                              # (3072,)
    top_i = np.argsort(np.abs(w_e))[-TOP_K:][::-1]    # descending abs weight
    top_dims[emotion] = [(int(i), round(float(w_e[i]), 4)) for i in top_i]
    dim_str = ', '.join(f"d{i}({v:+.3f})" for i, v in top_dims[emotion][:5])
    print(f"  {emotion:12s}: {dim_str}")

# %%
# ── 4. Emotion direction cosine similarity ────────────────────────────────────
# How similar are two emotions' weight vectors?
# cos(w_e, w_f) = 1 → same direction (emotions share the same dimensions)
# cos(w_e, w_f) = 0 → orthogonal (emotions use different dimensions)
# cos(w_e, w_f) = -1 → opposite (high score for one suppresses the other)

print(f"\n── Cosine similarity between emotion directions (layer {BEST_LAYER}) ──")
W_norm = W_best / (np.linalg.norm(W_best, axis=1, keepdims=True) + 1e-8)
cos_sim = W_norm @ W_norm.T   # (11, 11)

header = f"{'':12s}" + ''.join(f"{e[:7]:>9}" for e in EMOTIONS)
print(header)
for i, ei in enumerate(EMOTIONS):
    row = f"{ei:12s}" + ''.join(f"{cos_sim[i,j]:>9.3f}" for j in range(len(EMOTIONS)))
    print(row)

# %%
# ── 5. Dimension overlap between emotions ────────────────────────────────────
# How many of the top-k dimensions are shared between each pair of emotions?

print(f"\n── Top-{TOP_K} dimension overlap between emotion pairs (layer {BEST_LAYER}) ──")
top_sets = {e: set(d for d, _ in top_dims[e]) for e in EMOTIONS}
header = f"{'':12s}" + ''.join(f"{e[:7]:>9}" for e in EMOTIONS)
print(header)
for ei in EMOTIONS:
    row = f"{ei:12s}"
    for ej in EMOTIONS:
        overlap = len(top_sets[ei] & top_sets[ej])
        row += f"{overlap:>9}"
    print(row)

# %%
# ── Save summary ──────────────────────────────────────────────────────────────
summary = {
    'best_layer': BEST_LAYER,
    'emotions': EMOTIONS,
    'norms_per_layer': {
        L: {e: round(float(norms[L][i]), 4) for i, e in enumerate(EMOTIONS)}
        for L in sorted(norms.keys())
    },
    'weight_variance_per_layer': {
        L: round(float(layer_weights[L].var(axis=1).mean()), 8)
        for L in sorted(layer_weights.keys())
    },
    'top_dims_per_emotion': top_dims,
    'cosine_similarity': {
        EMOTIONS[i]: {EMOTIONS[j]: round(float(cos_sim[i, j]), 4)
                      for j in range(len(EMOTIONS))}
        for i in range(len(EMOTIONS))
    },
}

out_path = STEP5_PATH / 'probe_weight_analysis.json'
with open(out_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved: {out_path}")
