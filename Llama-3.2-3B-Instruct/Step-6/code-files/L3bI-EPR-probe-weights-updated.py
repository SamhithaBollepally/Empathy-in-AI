# %%
# ============================================================================
# Step 6 — Probe Weight Analysis
#
# Analyses the probe weight matrix W (10 x 3072) at every layer to answer:
#   1. How large is each emotion's weight vector? (norm per emotion per layer)
#   2. How much do weights vary across dimensions? (variance per layer)
#   3. Which dimensions matter most per emotion? (top-k dims per emotion)
#   4. How similar are emotion directions to each other? (cosine similarity)
#   5. How does weight structure change across layers?
#
# Which probe each section uses:
#   Sections 1, 2, 4 -> dense L2 weights. Norms and direction cosines are
#       meaningful only when all 3072 weights are live.
#   Sections 3, 5    -> sparse 1-SE L1 weights. "Which dims matter" needs real
#       feature selection; the L1 probe zeroes ~60% of weights while staying
#       within 1 std of the best accuracy, so its nonzero entries are the dims
#       the optimiser actually chose to keep. On dense L2 "top-k" is just the
#       largest of 3072 nonzero numbers - not a selection.
#
# CAVEAT for the write-up: probe weights are correlational, not causal. A large
# weight on dimension d means d is a reliable predictor, not that the model
# "stores" the emotion there. Causality comes from interventions (steering,
# ablation), not from this file.
#
# No model or GPU required — runs on saved emotion_probes.pt only.
# ============================================================================

# %%
import json
from pathlib import Path

import numpy as np
import torch

# %%
# ── Config ───────────────────────────────────────────────────────────────────
BASE_PATH = Path('/Volumes/Reading/Projects/Empathy-LLM/Llama-3.2-3B-Instruct/')
# BASE_PATH = Path('/content/drive/MyDrive/Project-4/')

DATA_PATH = BASE_PATH / 'Step-6' / 'data'

TOP_K = 10   # top dimensions to report per emotion

# %%
# ── Load probe bundle (new format: plain tensors W, b, mu, sd per layer) ──────
bundle = torch.load(DATA_PATH / 'emotion_probes.pt', map_location='cpu', weights_only=False)
EMOTIONS   = [str(c) for c in bundle['classes']]   # 10 emotion labels
N_LAYERS   = len(bundle['layers'])                  # 0..28
HIDDEN_DIM = bundle['hidden_dim']                   # 3072
BEST_LAYER = bundle['best_layer_l2']                # chosen by CV, not hardcoded
print(f"Emotions : {EMOTIONS}")
print(f"Layers   : {N_LAYERS}  (0 = embeddings, 1-{N_LAYERS - 1} = decoder)")
print(f"Hidden   : {HIDDEN_DIM}")
print(f"Best L2 layer (by CV): {BEST_LAYER}")

# %%
# ── Extract W per layer ───────────────────────────────────────────────────────
# Bundle stores W as (hidden_dim, num_classes) = (3072, 10); transpose to the
# (num_classes, hidden_dim) layout this analysis uses: row e = recipe for emotion e.
layer_weights_l2 = {}    # dense ridge weights
layer_weights_l1 = {}    # sparse 1-SE lasso weights (exact zeros)
for L in range(N_LAYERS):
    if L not in bundle['layers']:
        continue
    layer_weights_l2[L] = bundle['layers'][L]['l2']['W'].numpy().T       # (10, 3072)
    layer_weights_l1[L] = bundle['layers'][L]['l1_1se']['W'].numpy().T   # (10, 3072)

sparsity = {L: float((W == 0).mean()) for L, W in layer_weights_l1.items()}
print(f"Loaded L2 + sparse-L1 weights for {len(layer_weights_l2)} layers.")
print(f"Sparse-L1 zeros at best layer {BEST_LAYER}: {sparsity[BEST_LAYER]:.1%} "
      f"({int((1 - sparsity[BEST_LAYER]) * HIDDEN_DIM * len(EMOTIONS))} nonzero "
      f"of {HIDDEN_DIM * len(EMOTIONS)})")

# %%
# ── 1. Emotion weight norms per layer (L2) ────────────────────────────────────
# ||w_e||  for each emotion e at each layer
# A larger norm means the probe relies more heavily on the hidden-state
# dimensions collectively to predict emotion e at that layer.

norms = {}   # norms[L] = (10,) array
for L, W in layer_weights_l2.items():
    norms[L] = np.linalg.norm(W, axis=1)   # (10,)

# Print: for each layer, which emotion has the highest norm?
print("\n── Emotion weight norms per layer (top emotion) ──")
print(f"{'Layer':>6}  {'Top emotion':>12}  {'Norm':>8}  {'Min norm':>9}  {'Max norm':>9}  {'Variance':>10}")
for L in sorted(norms.keys()):
    n   = norms[L]
    top = EMOTIONS[np.argmax(n)]
    print(f"{L:>6}  {top:>12}  {np.max(n):>8.4f}  {np.min(n):>9.4f}  {np.max(n):>9.4f}  {n.var():>10.6f}")

# %%
# ── 2. Weight variance across dimensions per layer (L2) ───────────────────────
# var(W_L) averaged across emotions.
# High variance = probe uses a diverse set of dimensions.
# Low variance  = weights are concentrated near zero (uniform uncertainty).

print("\n── Mean weight variance across all dimensions per layer ──")
print(f"{'Layer':>6}  {'Mean var (across dims & emotions)':>35}  {'Sparse-L1 zeros':>15}")
for L in sorted(layer_weights_l2.keys()):
    W       = layer_weights_l2[L]              # (10, 3072)
    var_per_emotion = W.var(axis=1)            # (10,)  variance over 3072 dims
    mean_var = var_per_emotion.mean()
    print(f"{L:>6}  {mean_var:>35.8f}  {sparsity[L]:>14.1%}")

# %%
# ── 3. Top-k dimensions per emotion (sparse 1-SE L1, at the best L1 layer) ────
# Sparse probe: nonzero weights are a real selection. We list the top-k by
# |weight| among NONZERO entries only, plus each emotion's total nonzero count.

BEST_L1_LAYER = bundle['best_layer_l1']
W_sparse = layer_weights_l1[BEST_L1_LAYER]     # (10, 3072)

print(f"\n── Top-{TOP_K} nonzero dimensions per emotion (sparse L1, layer {BEST_L1_LAYER}) ──")
top_dims = {}
n_nonzero = {}
for e_idx, emotion in enumerate(EMOTIONS):
    w_e      = W_sparse[e_idx]                                   # (3072,)
    nz       = np.flatnonzero(w_e)
    top_i    = nz[np.argsort(-np.abs(w_e[nz]))[:TOP_K]]          # largest |w| among nonzeros
    top_dims[emotion] = [(int(i), round(float(w_e[i]), 4)) for i in top_i]
    n_nonzero[emotion] = int(nz.size)
    dim_str = ', '.join(f"d{i}({v:+.3f})" for i, v in top_dims[emotion][:5])
    print(f"  {emotion:12s} ({nz.size:4d} nonzero): {dim_str}")

# %%
# ── 4. Emotion direction cosine similarity (L2, best L2 layer) ───────────────
# How similar are two emotions' weight vectors?
# cos(w_e, w_f) = 1 → same direction (emotions share the same dimensions)
# cos(w_e, w_f) = 0 → orthogonal (emotions use different dimensions)
# cos(w_e, w_f) = -1 → opposite (high score for one suppresses the other)

W_best = layer_weights_l2[BEST_LAYER]   # (10, 3072)

print(f"\n── Cosine similarity between emotion directions (L2, layer {BEST_LAYER}) ──")
W_norm = W_best / (np.linalg.norm(W_best, axis=1, keepdims=True) + 1e-8)
cos_sim = W_norm @ W_norm.T   # (10, 10)

header = f"{'':12s}" + ''.join(f"{e[:7]:>9}" for e in EMOTIONS)
print(header)
for i, ei in enumerate(EMOTIONS):
    row = f"{ei:12s}" + ''.join(f"{cos_sim[i,j]:>9.3f}" for j in range(len(EMOTIONS)))
    print(row)

# Cross-check vs the pair-probe result: the two non-distinguishable pairs should
# have the most positive weight-direction cosine.
off = ~np.eye(len(EMOTIONS), dtype=bool)
print("\nMost similar direction pairs "
      "(expect afraid/terrified and angry/furious on top if weights agree with "
      "the separability analysis):")
ii, jj = np.where(np.triu(np.ones_like(cos_sim, dtype=bool), 1))
order = np.argsort(-cos_sim[ii, jj])
for rank in order[:5]:
    print(f"  {EMOTIONS[ii[rank]]:11s} ~ {EMOTIONS[jj[rank]]:11s} "
          f"cos {cos_sim[ii[rank], jj[rank]]:+.3f}")

# %%
# ── 5. Dimension overlap between emotions (sparse 1-SE L1) ───────────────────
# On the sparse probe, sharing a nonzero dimension means both emotions were
# selected onto the same feature. Overlap counts are meaningful here; on dense
# L2 (all dims nonzero) an overlap count would just be a coincidence count.

print(f"\n── Top-{TOP_K} dimension overlap between emotion pairs "
      f"(sparse L1, layer {BEST_L1_LAYER}) ──")
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
    'best_layer_l2': BEST_LAYER,
    'best_layer_l1': BEST_L1_LAYER,
    'emotions': EMOTIONS,
    'weights_source': {'sections_1_2_4': 'l2 (dense ridge)',
                       'sections_3_5': 'l1_1se (sparse lasso, 1-SE rule)'},
    'norms_per_layer': {
        L: {e: round(float(norms[L][i]), 4) for i, e in enumerate(EMOTIONS)}
        for L in sorted(norms.keys())
    },
    'weight_variance_per_layer': {
        L: round(float(layer_weights_l2[L].var(axis=1).mean()), 8)
        for L in sorted(layer_weights_l2.keys())
    },
    'sparse_l1_zeros_per_layer': {
        L: round(sparsity[L], 4) for L in sorted(sparsity.keys())
    },
    'top_dims_per_emotion': top_dims,
    'nonzero_dims_per_emotion': n_nonzero,
    'cosine_similarity': {
        EMOTIONS[i]: {EMOTIONS[j]: round(float(cos_sim[i, j]), 4)
                      for j in range(len(EMOTIONS))}
        for i in range(len(EMOTIONS))
    },
}

out_path = DATA_PATH / 'probe_weight_analysis.json'
with open(out_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved: {out_path}")
