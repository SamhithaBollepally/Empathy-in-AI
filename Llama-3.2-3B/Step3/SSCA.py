# pip install cca-zoo
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

BASE_PATH = '/Volumes/Reading/Projects/Empathy-LLM/'



EMOTIONS = ['afraid', 'angry', 'anxious', 'devastated', 'lonely', 'sad',
            'terrified', 'furious', 'grateful', 'hopeful', 'faithful']

LAYERS_TO_TEST = list(range(29))

# ── Load prompt embeddings ───────────────────────────────────────────────────
print("Loading embeddings...")
with open(BASE_PATH + 'prompt_embeddings.json', 'r') as f:
    raw_prompt = json.load(f)

prompt_by_layer = defaultdict(list)   # layer → (1100, 3072)
prompt_labels   = []
prompt_indices  = []  # Track original indices

for context in EMOTIONS:
    for pidx in sorted(raw_prompt[context].keys(), key=int):
        for layer in LAYERS_TO_TEST:
            prompt_by_layer[layer].append(
                np.array(raw_prompt[context][pidx][str(layer)])
            )
        prompt_labels.append(context)
        prompt_indices.append((context, int(pidx)))

prompt_labels = np.array(prompt_labels)

# ── Load utterance embeddings ────────────────────────────────────────────────
utt_npz = np.load(BASE_PATH + 'utterance_embeddings_all_layers.npz', allow_pickle=True)

utt_by_layer  = defaultdict(list)
utt_labels    = []
utt_indices   = []  # Track original indices

response_df = pd.read_csv(BASE_PATH + 'response_prompts.csv')
for idx, row in response_df.iterrows():
    context = row['context']
    for layer in LAYERS_TO_TEST:
        key = f"{context}_{idx}_{layer}"
        if key in utt_npz:
            utt_by_layer[layer].append(utt_npz[key])
    utt_labels.append(context)
    utt_indices.append((context, idx))

utt_labels = np.array(utt_labels)
print(f"✓ Loaded {len(prompt_labels)} prompt samples, {len(utt_labels)} utterance samples")

# Verify dimensions
for layer in [0, 2, 28]:
    P_shape = np.array(prompt_by_layer[layer]).shape
    U_shape = np.array(utt_by_layer[layer]).shape
    print(f"  Layer {layer}: Prompts {P_shape}, Utterances {U_shape}")

# ── SCCA via PLS Canonical (practical for n<<p) ───────────────────────────────
from sklearn.cross_decomposition import PLSCanonical

N_COMPONENTS = 10   # number of latent dimensions to extract
results = {}

print("\n" + "="*80)
print("SPARSE CANONICAL CORRELATION ANALYSIS (SCCA)")
print("X = Prompt Embeddings (1100 × 3072)")
print("Y = Utterance Embeddings (1100 × 3072)")
print("="*80)

for layer in LAYERS_TO_TEST:
    print(f"\nLayer {layer}:")

    P = np.array(prompt_by_layer[layer])    # (1100, 3072)
    U = np.array(utt_by_layer[layer])       # (1100, 3072)

    # Verify dimensions
    if len(P) != len(U):
        print(f"  ⚠ Warning: Size mismatch - P:{len(P)}, U:{len(U)}")
    
    # Align sizes (in case of mismatch)
    n = min(len(P), len(U))
    P, U = P[:n], U[:n]
    labels = prompt_labels[:n]
    
    print(f"  Shapes: P={P.shape}, U={U.shape}")

    # Standardize
    P = StandardScaler().fit_transform(P)
    U = StandardScaler().fit_transform(U)

    # Reduce with PCA first (necessary: 3072 dims → 200 dims before SCCA)
    pca_p = PCA(n_components=200, random_state=42).fit_transform(P)
    pca_u = PCA(n_components=200, random_state=42).fit_transform(U)

    # PLS Canonical (sparse-like via regularization internally)
    plsc = PLSCanonical(n_components=N_COMPONENTS, max_iter=1000)
    plsc.fit(pca_p, pca_u)

    P_scores, U_scores = plsc.transform(pca_p, pca_u)  # (n, N_COMPONENTS)

    # Canonical correlations per component
    canon_corrs = [
        np.corrcoef(P_scores[:, i], U_scores[:, i])[0, 1]
        for i in range(N_COMPONENTS)
    ]
    print(f"  Canonical correlations (top 5): {[f'{c:.3f}' for c in canon_corrs[:5]]}")
    print(f"  Mean canonical correlation: {np.mean(canon_corrs):.4f}")

    # Emotion discrimination in canonical space
    # Build 11×11 similarity matrix from first 2 canonical variates
    P_canon_mean = {e: P_scores[labels==e].mean(axis=0) for e in EMOTIONS}
    U_canon_mean = {e: U_scores[labels==e].mean(axis=0) for e in EMOTIONS}

    P_mat = np.stack([P_canon_mean[e] for e in EMOTIONS])
    U_mat = np.stack([U_canon_mean[e] for e in EMOTIONS])

    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    matrix = cos_sim(U_mat, P_mat)

    within  = np.diag(matrix).mean()
    mask    = ~np.eye(11, dtype=bool)
    between = matrix[mask].mean()
    gap     = within - between

    print(f"  Within-class:  {within:.4f}")
    print(f"  Cross-class:   {between:.4f}")
    print(f"  Discrimination gap: {gap:.4f}")

    results[layer] = {
        'canon_corrs': canon_corrs,
        'within': within,
        'between': between,
        'gap': gap,
        'matrix': matrix
    }

# ── Plot discrimination gaps across layers ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

gaps   = [results[l]['gap']    for l in LAYERS_TO_TEST]
corrs  = [results[l]['canon_corrs'][0] for l in LAYERS_TO_TEST]

axes[0].plot(LAYERS_TO_TEST, gaps, 'b-o', markersize=4, linewidth=2)
axes[0].set_xlabel('Layer', fontsize=11)
axes[0].set_ylabel('Discrimination Gap', fontsize=11)
axes[0].set_title('SCCA: Discrimination Gap (Within − Between)', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(0, color='red', linestyle='--', alpha=0.5)

axes[1].plot(LAYERS_TO_TEST, corrs, 'r-o', markersize=4, linewidth=2)
axes[1].set_xlabel('Layer', fontsize=11)
axes[1].set_ylabel('Canonical Correlation', fontsize=11)
axes[1].set_title('First Canonical Correlation', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig(BASE_PATH + 'scca_results.png', dpi=300)
print("\n✓ Saved scca_results.png")

# ── Save stats ───────────────────────────────────────────────────────────────
stats = pd.DataFrame([{
    'layer': l,
    'canon_corr_1': results[l]['canon_corrs'][0],
    'within': results[l]['within'],
    'between': results[l]['between'],
    'gap': results[l]['gap']
} for l in LAYERS_TO_TEST])

stats.to_csv(BASE_PATH + 'scca_stats.csv', index=False)
print("✓ Saved scca_stats.csv")

print("\n" + "="*80)
print("SCCA RESULTS SUMMARY")
print("="*80)
print(stats.to_string(index=False))

# Find best layers
best_gap_layer = stats.loc[stats['gap'].idxmax()]
best_corr_layer = stats.loc[stats['canon_corr_1'].idxmax()]

print("\n" + "="*80)
print("BEST LAYERS:")
print("="*80)
print(f"Best discrimination gap: Layer {int(best_gap_layer['layer'])} (gap={best_gap_layer['gap']:.4f})")
print(f"Best canonical correlation: Layer {int(best_corr_layer['layer'])} (corr={best_corr_layer['canon_corr_1']:.4f})")
print("\n✓ SCCA analysis complete!")