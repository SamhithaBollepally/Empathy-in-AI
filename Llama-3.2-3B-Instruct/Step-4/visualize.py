# %%
# ============================================================================
# Presentation figures for the empathy-steering study. Reads only saved JSONs
# (no model needed). Existing Step-2 figures (per-layer probe accuracy, context
# separation, SCCA) are reused; this script builds the results figures:
#   1. Zeroshot vs CoT (expressiveness, self-intensity, probe accuracy)
#   2. Probe accuracy: prompts vs responses (transfer drop)
#   3. Per-emotion recall (zeroshot, L2)
#   4. Confusion matrix (zeroshot, L2)
#   5. Steering grid: L2 target accuracy (steer layer x coeff)
#   6. Steering grid: judged expressiveness (steer layer x coeff)
# ============================================================================
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
BASE_PATH = '/Volumes/Reading/Projects/Empathy-LLM/Llama-3.2-3B-Instruct/'
#BASE_PATH = '/content/drive/MyDrive/Project-4/'

EMOTIONS = ['afraid', 'angry', 'anxious', 'devastated', 'lonely', 'sad',
            'terrified', 'furious', 'grateful', 'hopeful', 'faithful']
CHANCE = 1 / len(EMOTIONS)

STEP2 = BASE_PATH + 'Step-2/'
STEP3 = BASE_PATH + 'Step-3/'
STEP4 = BASE_PATH + 'Step-4/'
OUTPUTS = BASE_PATH + 'Outputs/'
FIG_DIR = STEP4 + 'figures/'
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', context='talk')

def load(path):
    with open(path) as f:
        return json.load(f)

def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR + name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: figures/{name}")

# %%
# ---- Gather numbers ----------------------------------------------------------
probe = load(STEP2 + 'probing_results.json')
PROMPT_L2 = max(probe['l2_accuracies'].values())
PROMPT_L1 = max(probe['l1_accuracies'].values())

def resp_stats(prefix):
    js = si = None
    d = load(OUTPUTS + f'{prefix}_responses.json')
    flat = [it for e in EMOTIONS for it in d[e]]
    js = np.mean([it['judged_score'] for it in flat if it.get('judged_score') is not None])
    si = np.mean([it['self_intensity'] for it in flat if it.get('self_intensity') is not None])
    l1 = load(STEP3 + f'{prefix}_emotion_predictions_l1.json')['summary']['accuracy']
    l2 = load(STEP3 + f'{prefix}_emotion_predictions_l2.json')['summary']['accuracy']
    return dict(judged=js, self_int=si, l1=l1, l2=l2)

zs, cot = resp_stats('zeroshot'), resp_stats('CoT')

# %%
# ---- Fig 1: Zeroshot vs CoT --------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
labels = ['Zeroshot', 'CoT']
x = np.arange(2)
w = 0.35
a1.bar(x - w/2, [zs['judged'], cot['judged']], w, label='Judged expressiveness', color='#4C72B0')
a1.bar(x + w/2, [zs['self_int'], cot['self_int']], w, label='Self-intensity', color='#DD8452')
a1.set_xticks(x); a1.set_xticklabels(labels); a1.set_ylim(0, 5)
a1.set_ylabel('Mean rating (1-5)'); a1.set_title('Response emotionality')
a1.legend(fontsize=12)
for i, v in enumerate([zs['judged'], cot['judged']]): a1.text(i - w/2, v + .05, f'{v:.2f}', ha='center', fontsize=11)
for i, v in enumerate([zs['self_int'], cot['self_int']]): a1.text(i + w/2, v + .05, f'{v:.2f}', ha='center', fontsize=11)

a2.bar(x - w/2, [zs['l2'], cot['l2']], w, label='L2 probe', color='#55A868')
a2.bar(x + w/2, [zs['l1'], cot['l1']], w, label='L1 probe', color='#C44E52')
a2.axhline(CHANCE, ls=':', color='gray', label=f'Chance ({CHANCE:.2f})')
a2.set_xticks(x); a2.set_xticklabels(labels); a2.set_ylim(0, 1)
a2.set_ylabel('Target-emotion accuracy'); a2.set_title('Probe recovers emotion from response')
a2.legend(fontsize=12)
for i, v in enumerate([zs['l2'], cot['l2']]): a2.text(i - w/2, v + .02, f'{v:.2f}', ha='center', fontsize=11)
for i, v in enumerate([zs['l1'], cot['l1']]): a2.text(i + w/2, v + .02, f'{v:.2f}', ha='center', fontsize=11)
save(fig, 'fig1_zeroshot_vs_cot.png')

# %%
# ---- Fig 2: Probe accuracy prompts vs responses (transfer) -------------------
fig, ax = plt.subplots(figsize=(9, 6))
names = ['Prompts\n(held-out)', 'Zeroshot\nresponses', 'CoT\nresponses']
vals = [PROMPT_L2, zs['l2'], cot['l2']]
bars = ax.bar(names, vals, color=['#4C72B0', '#55A868', '#8172B3'])
ax.axhline(CHANCE, ls=':', color='gray', label=f'Chance ({CHANCE:.2f})')
ax.set_ylim(0, 1); ax.set_ylabel('L2 probe accuracy')
ax.set_title('Emotion probe: prompt -> response transfer')
for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width()/2, v + .02, f'{v:.2f}', ha='center', fontsize=13)
ax.legend()
save(fig, 'fig2_probe_transfer.png')

# %%
# ---- Fig 3 & 4: per-emotion recall + confusion matrix (zeroshot L2) ----------
pred = load(STEP3 + 'zeroshot_emotion_predictions_l2.json')['predictions']
idx = {e: i for i, e in enumerate(EMOTIONS)}
M = np.zeros((len(EMOTIONS), len(EMOTIONS)))
for true_e, recs in pred.items():
    for r in recs:
        p = r['predicted_emotion']
        if p != 'Null':
            M[idx[true_e], idx[p]] += 1
row_sums = M.sum(axis=1, keepdims=True)
recall = np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums != 0)
diag = np.diag(recall)

# Fig 3: sorted recall bars
order = np.argsort(diag)[::-1]
fig, ax = plt.subplots(figsize=(12, 6))
colors = ['#55A868' if diag[i] >= 0.5 else '#C44E52' for i in order]
ax.bar(np.arange(len(order)), diag[order], color=colors)
ax.axhline(CHANCE, ls=':', color='gray', label=f'Chance ({CHANCE:.2f})')
ax.set_ylim(0, 1); ax.set_ylabel('Recall'); ax.set_title('Per-emotion recall (zeroshot, L2 probe)')
ax.set_xticks(np.arange(len(order)))
ax.set_xticklabels([EMOTIONS[i] for i in order], rotation=45, ha='right')
for j, i in enumerate(order): ax.text(j, diag[i] + .02, f'{diag[i]:.2f}', ha='center', fontsize=11)
ax.legend()
save(fig, 'fig3_per_emotion_recall.png')

# Fig 4: confusion matrix
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(recall, xticklabels=EMOTIONS, yticklabels=EMOTIONS, annot=True, fmt='.2f',
            cmap='magma', vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Recall'})
ax.set_xlabel('Predicted'); ax.set_ylabel('True emotion')
ax.set_title('Confusion matrix (zeroshot, L2 probe)')
save(fig, 'fig4_confusion_matrix.png')

# %%
# ---- Fig 5 & 6: steering grids (steer layer x coeff) -------------------------
steer = load(STEP4 + 'steering_probe_eval.json')
base_l2 = next(r['target_acc_l2'] for r in steer if r['output_file'] == 'baseline')
rows = [r for r in steer if r['output_file'] != 'baseline']
layers = sorted({r['layer_steered'] for r in rows})
coeffs = sorted({r['coeff'] for r in rows})

def grid(rows, key):
    G = np.full((len(layers), len(coeffs)), np.nan)
    li = {l: i for i, l in enumerate(layers)}; ci = {c: i for i, c in enumerate(coeffs)}
    for r in rows:
        if r.get(key) is not None:
            G[li[r['layer_steered']], ci[r['coeff']]] = r[key]
    return G

G_l2 = grid(rows, 'target_acc_l2')
fig, ax = plt.subplots(figsize=(9, 8))
sns.heatmap(G_l2, xticklabels=coeffs, yticklabels=layers, annot=True, fmt='.2f',
            cmap='RdBu_r', center=base_l2, ax=ax, cbar_kws={'label': 'L2 target accuracy'})
ax.set_xlabel('Steering coefficient'); ax.set_ylabel('Steered layer')
ax.set_title(f'Steering: L2 target accuracy\n(centered at baseline {base_l2:.2f}; red=above, blue=below)')
save(fig, 'fig5_steering_l2_grid.png')

# Fig 6: judged expressiveness grid (from the generation sweep log)
sweep_path = OUTPUTS + 'Steering-initial/steering_sweep_log.json'
if os.path.exists(sweep_path):
    sweep = load(sweep_path)
    base_judged = 3.309  # unsteered zeroshot, first-20/emotion
    G_j = grid(sweep, 'judged_score_mean')
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(G_j, xticklabels=coeffs, yticklabels=layers, annot=True, fmt='.2f',
                cmap='RdBu_r', center=base_judged, ax=ax,
                cbar_kws={'label': 'Judged expressiveness (1-5)'})
    ax.set_xlabel('Steering coefficient'); ax.set_ylabel('Steered layer')
    ax.set_title(f'Steering: judged expressiveness\n(centered at baseline {base_judged:.2f})')
    save(fig, 'fig6_steering_judged_grid.png')

print('\nAll figures written to', FIG_DIR)
