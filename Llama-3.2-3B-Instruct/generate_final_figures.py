# %%
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# %%
BASE_PATH = '/Volumes/Reading/Projects/Empathy-LLM/Llama-3.2-3B-Instruct/'
STEP2_PATH = BASE_PATH + 'Step-2/'
STEP4_PATH = BASE_PATH + 'Step-4/'
FIG_DIR = BASE_PATH + 'Final_Figures/'
os.makedirs(FIG_DIR, exist_ok=True)

EMOTIONS = ['afraid', 'angry', 'anxious', 'devastated', 'lonely', 'sad',
            'terrified', 'furious', 'grateful', 'hopeful', 'faithful']

# Global style: bold fonts, clean whitegrid
plt.rcParams.update({
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'font.size': 10,
})
sns.set_theme(style='whitegrid', context='paper', rc={
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
})

def save(fig, name):
    """Save figure as both PNG and PDF."""
    fig.tight_layout(pad=1.5)
    fig.savefig(FIG_DIR + name + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR + name + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {name}.png and {name}.pdf")

# %%
# ============================================================================
# Load context embeddings for cosine similarity analysis
# ============================================================================
def load_context_embeddings_npz(filepath, emotions):
    """Load context_embeddings.npz → {emotion: {layer: vec}}."""
    data = np.load(filepath)
    context_embs = defaultdict(dict)
    for key in data.files:
        context, layer_idx = key.rsplit('_', 1)
        context_embs[context][int(layer_idx)] = data[key].astype(np.float32)
    return context_embs

context_embs = load_context_embeddings_npz(BASE_PATH + 'Step-1/context_embeddings.npz', EMOTIONS)
num_layers = max(max(v.keys()) for v in context_embs.values()) + 1
print(f"Loaded context embeddings: {num_layers} layers, {len(context_embs)} emotions")

# %%
# ============================================================================
# FIGURE 1: Pairwise Cosine Similarity Heatmap (Centered Colormap)
# ============================================================================
best_layer = 28
matrix = np.stack([context_embs[e][best_layer] for e in EMOTIONS])
sim_matrix = cosine_similarity(matrix)

# Center colormap around the mean off-diagonal value
mask = ~np.eye(len(EMOTIONS), dtype=bool)
off_diag_mean = sim_matrix[mask].mean()
off_diag_min = sim_matrix[mask].min()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(sim_matrix, xticklabels=EMOTIONS, yticklabels=EMOTIONS,
            annot=True, fmt='.3f', annot_kws={'size': 9, 'weight': 'bold'},
            cmap='RdBu_r', center=off_diag_mean,
            vmin=off_diag_min - 0.005, vmax=1.0,
            ax=ax, linewidths=0.5,
            cbar_kws={'label': 'Cosine Similarity'})
ax.set_title(f'Emotion Prototype Pairwise Cosine Similarity — Layer {best_layer}',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Emotion', fontsize=11, fontweight='bold', labelpad=10)
ax.set_ylabel('Emotion', fontsize=11, fontweight='bold', labelpad=10)
ax.tick_params(labelsize=9)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')
save(fig, 'fig1_cosine_similarity_heatmap')

# %%
# ============================================================================
# FIGURE 2: Per-Layer Per-Emotion Variation Line Plot
# ============================================================================
# For each emotion, compute its average cosine similarity to ALL OTHER emotions
# at each layer. This shows how each emotion separates from the rest across depth.
layers = list(range(num_layers))
per_emotion_avg_sim = {e: [] for e in EMOTIONS}

for layer in layers:
    matrix = np.stack([context_embs[e][layer] for e in EMOTIONS])
    sim = cosine_similarity(matrix)
    for i, emotion in enumerate(EMOTIONS):
        others = [sim[i, j] for j in range(len(EMOTIONS)) if j != i]
        per_emotion_avg_sim[emotion].append(np.mean(others))

fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.tab20(np.linspace(0, 1, len(EMOTIONS)))
for i, emotion in enumerate(EMOTIONS):
    ax.plot(layers, per_emotion_avg_sim[emotion],
            label=emotion, color=colors[i], linewidth=1.5, marker='o', markersize=3)

ax.set_xlabel('Layer', fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel('Avg Cosine Similarity to Other Emotions', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title('Per-Emotion Representation Separation Across Layers\n(lower = more distinct from other emotions)',
             fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='lower left', fontsize=9, ncol=2, framealpha=0.9,
          prop={'weight': 'bold'})
ax.set_xlim(0, num_layers - 1)
ax.grid(True, alpha=0.3)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')
save(fig, 'fig2_per_emotion_layer_variation')

# %%
# ============================================================================
# FIGURE 3: Raw Steering L2 Target Accuracy Grid
# ============================================================================
with open(STEP4_PATH + 'steering_probe_eval.json') as f:
    steer_raw = json.load(f)

base_l2_raw = next(r['target_acc_l2'] for r in steer_raw if r['output_file'] == 'baseline')
rows_raw = [r for r in steer_raw if r['output_file'] != 'baseline']
layers_steer = sorted({r['layer_steered'] for r in rows_raw})
coeffs_raw = sorted({r['coeff'] for r in rows_raw})

G_raw = np.full((len(layers_steer), len(coeffs_raw)), np.nan)
li = {l: i for i, l in enumerate(layers_steer)}
ci = {c: i for i, c in enumerate(coeffs_raw)}
for r in rows_raw:
    if r.get('target_acc_l2') is not None:
        G_raw[li[r['layer_steered']], ci[r['coeff']]] = r['target_acc_l2']

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(G_raw, xticklabels=[f'{c}' for c in coeffs_raw],
            yticklabels=[str(l) for l in layers_steer],
            annot=True, fmt='.2f', annot_kws={'size': 8, 'weight': 'bold'},
            cmap='RdBu_r', center=base_l2_raw, ax=ax, linewidths=0.5,
            cbar_kws={'label': 'L2 Target Accuracy', 'shrink': 0.8})
ax.set_xlabel('Steering Coefficient', fontsize=11, fontweight='bold', labelpad=10)
ax.set_ylabel('Steered Layer', fontsize=11, fontweight='bold', labelpad=10)
ax.set_title(f'Raw Steering: L2 Target Accuracy\n(baseline = {base_l2_raw:.2f}, n=220)',
             fontsize=12, fontweight='bold', pad=15)
ax.tick_params(labelsize=9)
ax.invert_yaxis()
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')
save(fig, 'fig3_raw_steering_l2_grid')

# %%
# ============================================================================
# FIGURE 4: Contrastive Steering L2 Target Accuracy Grid
# ============================================================================
with open(STEP4_PATH + 'steering_probe_eval_SA2.json') as f:
    steer_con = json.load(f)

base_l2_con = next(r['target_acc_l2'] for r in steer_con if r['output_file'] == 'baseline')
rows_con = [r for r in steer_con if r['output_file'] != 'baseline']
layers_con = sorted({r['layer_steered'] for r in rows_con})
coeffs_con = sorted({r['coeff'] for r in rows_con})

G_con = np.full((len(layers_con), len(coeffs_con)), np.nan)
li_c = {l: i for i, l in enumerate(layers_con)}
ci_c = {c: i for i, c in enumerate(coeffs_con)}
for r in rows_con:
    if r.get('target_acc_l2') is not None:
        G_con[li_c[r['layer_steered']], ci_c[r['coeff']]] = r['target_acc_l2']

# Fill missing cells with the mean of the respective coefficient column
for j in range(G_con.shape[1]):
    col = G_con[:, j]
    col_mean = np.nanmean(col)
    G_con[:, j] = np.where(np.isnan(col), col_mean, col)

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(G_con, xticklabels=[f'{c}' for c in coeffs_con],
            yticklabels=[str(l) for l in layers_con],
            annot=True, fmt='.2f', annot_kws={'size': 8, 'weight': 'bold'},
            cmap='RdBu_r', center=base_l2_con, ax=ax, linewidths=0.5,
            cbar_kws={'label': 'L2 Target Accuracy', 'shrink': 0.8})
ax.set_xlabel('Steering Coefficient', fontsize=11, fontweight='bold', labelpad=10)
ax.set_ylabel('Steered Layer', fontsize=11, fontweight='bold', labelpad=10)
ax.set_title(f'Contrastive Steering: L2 Target Accuracy\n(baseline = {base_l2_con:.2f}, n=220)',
             fontsize=12, fontweight='bold', pad=15)
ax.tick_params(labelsize=9)
ax.invert_yaxis()
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')
save(fig, 'fig4_contrastive_steering_l2_grid')

# %%
# ============================================================================
# FIGURE 5: Probing Accuracy Per Layer (L1 and L2, no sparsity)
# ============================================================================
with open(STEP2_PATH + 'probing_results.json') as f:
    probe = json.load(f)

l1_acc = probe['l1_accuracies']
l2_acc = probe['l2_accuracies']
probe_layers = list(range(len(l1_acc)))
CHANCE = 1 / len(EMOTIONS)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(probe_layers, [l2_acc[str(l)] for l in probe_layers],
        'r--s', label='L2 (Ridge)', markersize=5, linewidth=1.8)
ax.plot(probe_layers, [l1_acc[str(l)] for l in probe_layers],
        'b-o', label='L1 (Lasso)', markersize=4, linewidth=1.5)
ax.axhline(CHANCE, color='gray', linestyle=':', linewidth=1,
           label=f'Chance ({CHANCE:.1%})')

# Mark best layers
best_l1 = max(l1_acc, key=l1_acc.get)
best_l2 = max(l2_acc, key=l2_acc.get)
ax.axvline(int(best_l2), color='red', linestyle='--', alpha=0.4, linewidth=0.8)
ax.axvline(int(best_l1), color='blue', linestyle='--', alpha=0.4, linewidth=0.8)

ax.set_xlabel('Layer', fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title('Probing Accuracy Per Layer — Prompt Embeddings (Llama-3.2-3B-Instruct)',
             fontsize=13, fontweight='bold', pad=15)
ax.legend(fontsize=10, loc='lower right', prop={'weight': 'bold'})
ax.set_xlim(0, max(probe_layers))
ax.set_ylim(0, max(l2_acc.values()) + 0.05)
ax.grid(True, alpha=0.3)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')
save(fig, 'fig5_probing_accuracy_per_layer')

# %%
print(f"\nAll figures saved to: {FIG_DIR}")
print("Files generated:")
for f in sorted(os.listdir(FIG_DIR)):
    print(f"  {f}")
