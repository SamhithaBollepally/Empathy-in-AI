# %%
import os
import json
import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# %%
BASE_PATH = '/Volumes/Reading/Projects/Empathy-LLM/Llama-3.2-3B-Instruct/'
#BASE_PATH = '/content/drive/MyDrive/Project-4/'

EMOTIONS = ['afraid', 'angry', 'anxious', 'devastated', 'lonely', 'sad',
            'terrified', 'furious', 'grateful', 'hopeful', 'faithful']

STEP1_PATH = BASE_PATH + 'Step-1/'
STEP2_PATH = BASE_PATH + 'Step-2/'
os.makedirs(STEP2_PATH, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# %%
def load_context_item_npz(filepath):
    """Load NPZ with keys '{context}_{item_idx}_{layer_idx}'.
    Returns: {context: ndarray (num_layers, num_items, hidden_dim)} as float32.
    Avoids forming any intermediate full matrix.
    """
    data = np.load(filepath)

    nested = defaultdict(lambda: defaultdict(dict))
    for key in data.files:
        parts = key.rsplit('_', 2)
        context, item_idx, layer_idx = parts[0], int(parts[1]), int(parts[2])
        nested[context][item_idx][layer_idx] = data[key].astype(np.float32)
    del data

    result = {}
    for context, items in nested.items():
        sample_layers = next(iter(items.values()))
        num_layers = len(sample_layers)
        hidden_dim = next(iter(sample_layers.values())).shape[0]
        num_items  = len(items)

        arr = np.empty((num_layers, num_items, hidden_dim), dtype=np.float32)
        for item_idx, layers in items.items():
            for layer_idx, emb in layers.items():
                arr[layer_idx, item_idx] = emb
        result[context] = arr

    del nested
    return result

# %%
print("Loading prompt embeddings (Set 1)...")
prompt_embs = load_context_item_npz(STEP1_PATH + 'prompt_embeddings.npz')
print("Loading utterance embeddings (Set 2)...")
utterance_embs = load_context_item_npz(STEP1_PATH + 'utterance_embeddings.npz')

num_layers = next(iter(prompt_embs.values())).shape[0]
print(f"  num_layers={num_layers} | contexts={list(prompt_embs.keys())}")

# %%
# P_mean[e] and U_mean[e]: mean embedding per emotion per layer → (num_layers, hidden_dim)
P_mean = {e: prompt_embs[e].mean(axis=1)    for e in EMOTIONS}
U_mean = {e: utterance_embs[e].mean(axis=1) for e in EMOTIONS}

# %%
def compute_pairwise_cosine(P_mean, U_mean, emotions, layer):
    """matrix[i,j] = cosine_sim(U_mean[i], P_mean[j]).
    Diagonal = within-emotion, off-diagonal = cross-emotion.
    """
    P_mat = np.stack([P_mean[e][layer] for e in emotions])   # (11, D)
    U_mat = np.stack([U_mean[e][layer] for e in emotions])   # (11, D)
    return cosine_similarity(U_mat, P_mat)                    # (11, 11)

# Find best layer: largest gap between diagonal (within) and off-diagonal (between)
cosine_gaps = {}
for layer in range(num_layers):
    mat       = compute_pairwise_cosine(P_mean, U_mean, EMOTIONS, layer)
    diag_mean = np.diag(mat).mean()
    off_mean  = mat[~np.eye(len(EMOTIONS), dtype=bool)].mean()
    cosine_gaps[layer] = diag_mean - off_mean

best_cosine_layer = max(cosine_gaps, key=cosine_gaps.get)
print(f"Best pairwise cosine layer: {best_cosine_layer} "
      f"(within={np.diag(compute_pairwise_cosine(P_mean, U_mean, EMOTIONS, best_cosine_layer)).mean():.4f}, "
      f"gap={cosine_gaps[best_cosine_layer]:.4f})")

# %%
def scca_torch(X, Y, lambda_u=0.1, lambda_v=0.1, n_iter=200):
    """Sparse CCA via iterative soft-thresholding (Witten et al. 2009).

    Avoids forming the (p × q) cross-covariance matrix — uses X^T(Yv)/n instead.
    X: (n, p), Y: (n, q) torch tensors on same device.
    Returns: canonical correlation, sparse u (p,), sparse v (q,).
    """
    n = X.shape[0]
    X = X - X.mean(0)
    Y = Y - Y.mean(0)

    v = torch.randn(Y.shape[1], device=X.device, dtype=torch.float32)
    v = v / v.norm()

    def soft_threshold(z, lam):
        return torch.sign(z) * torch.clamp(z.abs() - lam, min=0.0)

    for _ in range(n_iter):
        # u update: X^T (Y v) / n
        u_st = soft_threshold((X.T @ (Y @ v)) / n, lambda_u)
        u_n  = u_st.norm()
        u    = u_st / u_n if u_n > 1e-10 else u_st

        # v update: Y^T (X u) / n
        v_st = soft_threshold((Y.T @ (X @ u)) / n, lambda_v)
        v_n  = v_st.norm()
        v    = v_st / v_n if v_n > 1e-10 else v_st

    Xu = X @ u
    Yv = Y @ v
    corr = (Xu * Yv).sum() / (Xu.norm() * Yv.norm() + 1e-10)
    return corr.item(), u.cpu().numpy(), v.cpu().numpy()

# %%
def scca_torch_best(X, Y, lambda_u=0.009, lambda_v=0.009, n_iter=200, n_restarts=5):
    """Run SCCA with multiple random initialisations and return the best result."""
    best_corr, best_u, best_v = -1, None, None
    for seed in range(n_restarts):
        torch.manual_seed(seed)
        corr, u, v = scca_torch(X, Y, lambda_u, lambda_v, n_iter)
        if corr > best_corr:
            best_corr, best_u, best_v = corr, u, v
    return best_corr, best_u, best_v

# %%
# Within-context SCCA: Set 1 prompts ↔ Set 2 utterances, per emotion per layer
LAMBDA = 0.1

within_corr = {e: {} for e in EMOTIONS}

print("\nRunning within-context SCCA...")
for emotion in EMOTIONS:
    P = prompt_embs[emotion]      # (num_layers, 200, D)
    U = utterance_embs[emotion]   # (num_layers, 200, D)
    for layer in range(num_layers):
        X = torch.tensor(P[layer], dtype=torch.float32, device=device)
        Y = torch.tensor(U[layer], dtype=torch.float32, device=device)
        corr, _, _ = scca_torch_best(X, Y, lambda_u=LAMBDA, lambda_v=LAMBDA)
        within_corr[emotion][layer] = corr
    print(f"  {emotion} done — best layer corr: "
          f"{max(within_corr[emotion].values()):.4f} "
          f"@ layer {max(within_corr[emotion], key=within_corr[emotion].get)}")

# %%
# Between-context SCCA: all prompts vs all utterances stacked, per layer
between_corr = {}

print("\nRunning between-context SCCA...")
for layer in range(num_layers):
    X_all = np.concatenate([prompt_embs[e][layer]    for e in EMOTIONS], axis=0)  # (2200, D)
    Y_all = np.concatenate([utterance_embs[e][layer] for e in EMOTIONS], axis=0)  # (2200, D)
    X_t = torch.tensor(X_all, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y_all, dtype=torch.float32, device=device)
    corr, _, _ = scca_torch_best(X_t, Y_t, lambda_u=0.009, lambda_v=0.009)
    between_corr[layer] = corr
    print(f"  Layer {layer:2d} | between-context corr: {corr:.4f}")

# %%
# Save results
results = {
    'within_context_corr':  {e: {str(l): v for l, v in within_corr[e].items()} for e in EMOTIONS},
    'between_context_corr': {str(l): float(v) for l, v in between_corr.items()},
    'best_cosine_layer':    int(best_cosine_layer),
    'cosine_gaps':          {str(l): float(v) for l, v in cosine_gaps.items()},
}
with open(STEP2_PATH + 'scca_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved: scca_results.json")

# %%
# Plot 1: Pairwise cosine similarity heatmap at best layer
mat = compute_pairwise_cosine(P_mean, U_mean, EMOTIONS, best_cosine_layer)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(mat, xticklabels=EMOTIONS, yticklabels=EMOTIONS,
            annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax,
            linewidths=0.5)
ax.set_xlabel('Prompt Context — Set 1 (P_mean)')
ax.set_ylabel('Utterance Context — Set 2 (U_mean)')
ax.set_title(f'Pairwise Cosine Similarity: U_mean[i] vs P_mean[j] — Layer {best_cosine_layer}\n'
             f'Diagonal = within-emotion  |  Off-diagonal = cross-emotion')
plt.tight_layout()
plt.savefig(STEP2_PATH + 'scca_pairwise_cosine.png', dpi=300)
print("Saved: scca_pairwise_cosine.png")

# %%
# Plot 2: Within-context SCCA correlations across layers (one line per emotion)
layers = list(range(num_layers))
fig, ax = plt.subplots(figsize=(14, 5))
for emotion in EMOTIONS:
    ax.plot(layers, [within_corr[emotion][l] for l in layers],
            label=emotion, linewidth=1.5, marker='o', markersize=3)
ax.set_xlabel('Layer')
ax.set_ylabel('Canonical Correlation')
ax.set_title('Within-Context SCCA: Prompt–Utterance Alignment per Emotion per Layer')
ax.legend(loc='lower right', fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(STEP2_PATH + 'scca_within_context.png', dpi=300)
print("Saved: scca_within_context.png")

# %%
# Plot 3: Between-context SCCA correlation across layers
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(layers, [between_corr[l] for l in layers], 'm-o', markersize=4)
ax.set_xlabel('Layer')
ax.set_ylabel('Canonical Correlation')
ax.set_title('Between-Context SCCA: All Prompts vs All Utterances per Layer')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(STEP2_PATH + 'scca_between_context.png', dpi=300)
print("Saved: scca_between_context.png")
