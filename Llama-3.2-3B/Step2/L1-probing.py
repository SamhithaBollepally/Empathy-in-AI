import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

BASE_PATH = '/Volumes/Reading/Projects/Empathy-LLM/'

EMOTIONS = ['afraid', 'angry', 'anxious', 'devastated', 'lonely', 'sad',
            'terrified', 'furious', 'grateful', 'hopeful', 'faithful']

# Load embeddings
print("Loading prompt embeddings...")
with open(BASE_PATH + 'prompt_embeddings.json', 'r') as f:
    prompt_embs = json.load(f)

with open(BASE_PATH + 'context_prompts.json', 'r') as f:
    context_prompts = json.load(f)

num_layers = 29

# Build X, y arrays
X_all, y_all = [], []
for context in EMOTIONS:
    for prompt_idx_str in prompt_embs[context]:
        layers = prompt_embs[context][prompt_idx_str]
        layer_embs = [np.array(layers[str(l)]) for l in range(num_layers)]
        X_all.append(layer_embs)
        y_all.append(context)

X_all = np.array(X_all)   # (1100, 29, 3072)
y_all = np.array(y_all)

indices = np.arange(len(y_all))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_all)
y_train, y_test = y_all[train_idx], y_all[test_idx]

l1_accuracies = {}
l2_accuracies = {}
l1_sparsity   = {}     # fraction of zero coefficients per layer
top_features   = {}    # top active dimensions per emotion per layer

print("\nTraining L1 and L2 probes per layer...")
for layer in range(num_layers):
    X = X_all[:, layer, :]
    X_train, X_test = X[train_idx], X[test_idx]

    # L1 probe
    clf_l1 = LogisticRegression(penalty='l1', solver='saga', C=1.0,
                                 max_iter=2000, random_state=42, n_jobs=-1)
    clf_l1.fit(X_train, y_train)
    acc_l1 = accuracy_score(y_test, clf_l1.predict(X_test))
    l1_accuracies[layer] = acc_l1

    # Sparsity: fraction of zero weights across all emotion coefficients
    coef = clf_l1.coef_   # shape: (11, 3072)
    zero_frac = (coef == 0).mean()
    l1_sparsity[layer] = zero_frac

    # L2 probe (for comparison)
    clf_l2 = LogisticRegression(penalty='l2', solver='lbfgs', C=1.0,
                                 max_iter=1000, random_state=42, n_jobs=-1)
    clf_l2.fit(X_train, y_train)
    l2_accuracies[layer] = accuracy_score(y_test, clf_l2.predict(X_test))

    print(f"Layer {layer:2d} | L1: {acc_l1:.4f} | L2: {l2_accuracies[layer]:.4f} | Sparsity: {zero_frac:.2%}")

    # Save top active dimensions per emotion at best layer
    if layer == max(l1_accuracies, key=l1_accuracies.get):
        for i, emotion in enumerate(clf_l1.classes_):
            top_dims = np.argsort(np.abs(coef[i]))[-20:][::-1]
            top_features[emotion] = top_dims.tolist()

# Save results
results = {
    'l1_accuracies': {str(k): v for k, v in l1_accuracies.items()},
    'l2_accuracies': {str(k): v for k, v in l2_accuracies.items()},
    'l1_sparsity':   {str(k): v for k, v in l1_sparsity.items()},
    'top_features':  top_features
}
with open(BASE_PATH + 'l1_probing_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\n✓ Saved l1_probing_results.json")

# Plot L1 vs L2 accuracy across layers
layers = list(range(num_layers))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1.plot(layers, [l1_accuracies[l] for l in layers], 'b-o', label='L1 (Lasso)', markersize=4)
ax1.plot(layers, [l2_accuracies[l] for l in layers], 'r--s', label='L2 (Ridge)', markersize=4)
ax1.axhline(1/11, color='gray', linestyle=':', label='Chance (9.1%)')
ax1.set_ylabel('Accuracy')
ax1.set_title('L1 vs L2 Probing Accuracy Across Layers')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(layers, [l1_sparsity[l] for l in layers], 'g-o', markersize=4)
ax2.set_ylabel('Fraction of Zero Weights')
ax2.set_xlabel('Layer')
ax2.set_title('L1 Sparsity — How Many Dimensions Are Zeroed Out?')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(BASE_PATH + 'l1_vs_l2_probing.png', dpi=300)
print("✓ Saved l1_vs_l2_probing.png")