# %%
# ============================================================================
# Step 5m — Probe-Weight × Attention Joint Visualisation
#
# Combines two pre-computed analysis files:
#   • probe_weight_analysis.json        (from L3bI-EPR-probe-weights.py)
#   • attention_analysis_epr_transform_cubic_a0.0001.json  (from L3bI-EPR-attention-analysis.py)
#
# Produces a 6-panel figure:
#   1. Layer-wise mean probe norm vs mean emotional attention
#   2. Weight variance per layer vs mean emotional attention
#   3. Per-emotion probe norm (best layer) vs mean emotional attention
#   4. Cosine-similarity heatmap of emotion directions (probe weights)
#   5. Layer-wise emotional attention breakdown
#   6. Attention-head emotional-attention bar chart
# ============================================================================

# %%
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# %%
# ── Config ───────────────────────────────────────────────────────────────────
BASE_PATH  = Path('/Volumes/Reading/Projects/Empathy-LLM/Llama-3.2-3B-Instruct/')
STEP5_PATH = BASE_PATH / 'Step-5'

PROBE_FILE = STEP5_PATH / 'probe_weight_analysis.json'
ATTN_FILE  = STEP5_PATH / 'attention_analysis_epr_transform_cubic_a0.0001.json'
OUT_FILE   = STEP5_PATH / 'probe_attn_viz.png'

# %%
# ── Load data ─────────────────────────────────────────────────────────────────
with open(PROBE_FILE) as f:
    probe = json.load(f)

with open(ATTN_FILE) as f:
    att = json.load(f)

EMOTIONS   = probe['emotions']          # 11 labels, canonical order
BEST_LAYER = probe['best_layer']        # 15

# ── Probe data ────────────────────────────────────────────────────────────────
# norms_per_layer: dict[str(layer)] -> dict[emotion -> float]
probe_layers = sorted(probe['norms_per_layer'].keys(), key=int)   # '0' .. '28'
probe_layer_ids  = [int(L) for L in probe_layers]                 # 0..28

# Mean norm across emotions per layer
mean_probe_norm = np.array([
    np.mean([probe['norms_per_layer'][L][e] for e in EMOTIONS])
    for L in probe_layers
])

# Weight variance per layer
probe_var = np.array([probe['weight_variance_per_layer'][L] for L in probe_layers])

# Per-emotion norm at best layer
probe_norm_best = np.array([
    probe['norms_per_layer'][str(BEST_LAYER)][e] for e in EMOTIONS
])

# Cosine-similarity matrix at best layer
cos_sim = np.array([
    [probe['cosine_similarity'][ei][ej] for ej in EMOTIONS]
    for ei in EMOTIONS
])

# ── Attention data ─────────────────────────────────────────────────────────────
# layerwise arrays have 28 elements -> decoder layers 1..28
lm = att['summary']['layerwise_mean']
att_layers    = np.arange(1, 29)               # decoder layers 1..28
att_prompt    = np.array(lm['prompt_attention'])
att_generated = np.array(lm['generated_attention'])
att_emotional = np.array(lm['emotional_attention'])

# Per-emotion mean emotional attention (overall, not per-layer)
pe = att['summary']['per_emotion']
att_emo_per_emotion = np.array([pe[e]['mean_emotional_attention'] for e in EMOTIONS])

# Attention heads
head_emo = np.array(att['summary']['head_summary']['mean_emo_per_head'])   # (24,)

# ── Alignment: probe decoder layers 1..28 correspond to att_layers 1..28 ─────
# For overlay plots we restrict probe to layers 1..28 (drop layer 0 embeddings)
probe_decoder_mask = [i for i, L in enumerate(probe_layer_ids) if L >= 1]
probe_decoder_ids  = [probe_layer_ids[i] for i in probe_decoder_mask]
mean_probe_norm_dec = mean_probe_norm[probe_decoder_mask]   # (28,)
probe_var_dec       = probe_var[probe_decoder_mask]          # (28,)

# %%
# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = [
    '#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261',
    '#264653', '#A8DADC', '#6D6875', '#B5838D', '#E07A5F', '#3D405B',
]
emo_colors = {e: PALETTE[i] for i, e in enumerate(EMOTIONS)}

# %%
# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 18))
fig.patch.set_facecolor('#0f1117')

gs = gridspec.GridSpec(
    3, 2,
    figure=fig,
    hspace=0.45,
    wspace=0.35,
    left=0.07, right=0.97,
    top=0.93,  bottom=0.06,
)

AX_STYLE = dict(facecolor='#1a1d27', frameon=True)
TICK_KW   = dict(color='#aaaaaa', labelsize=9)
LABEL_KW  = dict(color='#cccccc', fontsize=10)
TITLE_KW  = dict(color='#ffffff', fontsize=11, fontweight='bold', pad=8)
GRID_KW   = dict(color='#2e3244', linewidth=0.6, linestyle='--', alpha=0.7)
SPINE_COL = '#3a3d50'

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(AX_STYLE['facecolor'])
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_COL)
    ax.tick_params(axis='both', **TICK_KW)
    ax.set_xlabel(xlabel, **LABEL_KW)
    ax.set_ylabel(ylabel, **LABEL_KW)
    ax.set_title(title, **TITLE_KW)
    ax.grid(**GRID_KW)
    ax.set_axisbelow(True)


# ── Panel 1: Mean probe norm vs emotional attention (layer-wise) ───────────────
ax1 = fig.add_subplot(gs[0, 0], **AX_STYLE)
style_ax(ax1,
         title='Panel 1 — Mean Probe Norm vs Emotional Attention (layer-wise)',
         xlabel='Layer', ylabel='Mean Probe Norm')

color_probe = '#7EB8F7'
color_attn  = '#F4A261'

ln1 = ax1.plot(probe_decoder_ids, mean_probe_norm_dec,
               color=color_probe, linewidth=2, marker='o', markersize=4,
               label='Mean probe norm')
ax1.axvline(BEST_LAYER, color='#aaaaaa', linewidth=1, linestyle=':', alpha=0.7)
ax1.text(BEST_LAYER + 0.3, ax1.get_ylim()[0], f'L{BEST_LAYER}',
         color='#aaaaaa', fontsize=8, va='bottom')

ax1b = ax1.twinx()
ax1b.set_facecolor('none')
ax1b.tick_params(axis='y', **TICK_KW)
ax1b.spines['right'].set_edgecolor(SPINE_COL)
ax1b.spines['left'].set_edgecolor(SPINE_COL)
ax1b.spines['top'].set_edgecolor(SPINE_COL)
ax1b.spines['bottom'].set_edgecolor(SPINE_COL)
ln2 = ax1b.plot(att_layers, att_emotional,
                color=color_attn, linewidth=2, marker='s', markersize=4,
                linestyle='--', label='Mean emotional attention')
ax1b.set_ylabel('Mean Emotional Attention', **LABEL_KW)

lines = ln1 + ln2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', facecolor='#1a1d27',
           edgecolor=SPINE_COL, labelcolor='#cccccc', fontsize=8)


# ── Panel 2: Weight variance per layer vs emotional attention ──────────────────
ax2 = fig.add_subplot(gs[0, 1], **AX_STYLE)
style_ax(ax2,
         title='Panel 2 — Probe Weight Variance vs Emotional Attention',
         xlabel='Layer', ylabel='Mean Weight Variance')

color_var = '#A8DADC'

ln3 = ax2.plot(probe_decoder_ids, probe_var_dec,
               color=color_var, linewidth=2, marker='o', markersize=4,
               label='Weight variance')
ax2.axvline(BEST_LAYER, color='#aaaaaa', linewidth=1, linestyle=':', alpha=0.7)

ax2b = ax2.twinx()
ax2b.set_facecolor('none')
ax2b.tick_params(axis='y', **TICK_KW)
for sp in ax2b.spines.values():
    sp.set_edgecolor(SPINE_COL)
ln4 = ax2b.plot(att_layers, att_emotional,
                color=color_attn, linewidth=2, marker='s', markersize=4,
                linestyle='--', label='Mean emotional attention')
ax2b.set_ylabel('Mean Emotional Attention', **LABEL_KW)

lines2 = ln3 + ln4
labels2 = [l.get_label() for l in lines2]
ax2.legend(lines2, labels2, loc='upper right', facecolor='#1a1d27',
           edgecolor=SPINE_COL, labelcolor='#cccccc', fontsize=8)


# ── Panel 3: Per-emotion probe norm vs emotional attention (best layer) ────────
ax3 = fig.add_subplot(gs[1, 0], **AX_STYLE)
style_ax(ax3,
         title=f'Panel 3 — Per-Emotion: Probe Norm (L{BEST_LAYER}) vs Emotional Attention',
         xlabel='Emotion', ylabel='Probe Weight Norm')

x = np.arange(len(EMOTIONS))
bar_w = 0.38

bars1 = ax3.bar(x - bar_w / 2, probe_norm_best,
                width=bar_w, color=[emo_colors[e] for e in EMOTIONS],
                alpha=0.85, label='Probe norm (L15)')

ax3b = ax3.twinx()
ax3b.set_facecolor('none')
ax3b.tick_params(axis='y', **TICK_KW)
for sp in ax3b.spines.values():
    sp.set_edgecolor(SPINE_COL)

bars2 = ax3b.bar(x + bar_w / 2, att_emo_per_emotion,
                 width=bar_w, color=[emo_colors[e] for e in EMOTIONS],
                 alpha=0.45, edgecolor='white', linewidth=0.5,
                 label='Emotional attention')
ax3b.set_ylabel('Mean Emotional Attention', **LABEL_KW)

ax3.set_xticks(x)
ax3.set_xticklabels(EMOTIONS, rotation=35, ha='right', fontsize=8, color='#cccccc')

from matplotlib.patches import Patch
legend_els = [
    Patch(facecolor='#888888', alpha=0.9, label='Probe norm (solid)'),
    Patch(facecolor='#888888', alpha=0.4, label='Emotional attention (transparent)'),
]
ax3.legend(handles=legend_els, loc='upper left', facecolor='#1a1d27',
           edgecolor=SPINE_COL, labelcolor='#cccccc', fontsize=8)


# ── Panel 4: Cosine similarity heatmap ────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1], **AX_STYLE)
style_ax(ax4,
         title=f'Panel 4 — Emotion Direction Cosine Similarity (probe L{BEST_LAYER})',
         xlabel='', ylabel='')

im = ax4.imshow(cos_sim, cmap='RdBu_r', vmin=-0.4, vmax=0.4, aspect='auto')
ax4.set_xticks(range(len(EMOTIONS)))
ax4.set_yticks(range(len(EMOTIONS)))
ax4.set_xticklabels(EMOTIONS, rotation=45, ha='right', fontsize=8, color='#cccccc')
ax4.set_yticklabels(EMOTIONS, fontsize=8, color='#cccccc')
ax4.grid(False)

for i in range(len(EMOTIONS)):
    for j in range(len(EMOTIONS)):
        val = cos_sim[i, j]
        txt_color = 'white' if abs(val) > 0.25 else '#888888'
        ax4.text(j, i, f'{val:.2f}', ha='center', va='center',
                 fontsize=6.5, color=txt_color)

cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
cbar.ax.tick_params(colors='#aaaaaa', labelsize=8)
cbar.outline.set_edgecolor(SPINE_COL)


# ── Panel 5: Layer-wise attention breakdown (stacked-style) ───────────────────
ax5 = fig.add_subplot(gs[2, 0], **AX_STYLE)
style_ax(ax5,
         title='Panel 5 — Layer-wise Attention Breakdown',
         xlabel='Layer', ylabel='Mean Attention')

ax5.plot(att_layers, att_prompt,    color='#7EB8F7', linewidth=2, label='Prompt attention')
ax5.plot(att_layers, att_generated, color='#F4A261', linewidth=2, label='Generated attention')
ax5.plot(att_layers, att_emotional, color='#2A9D8F', linewidth=2, marker='o', markersize=3,
         label='Emotional attention')

# highlight best probe layer
ax5.axvline(BEST_LAYER, color='#aaaaaa', linewidth=1, linestyle=':', alpha=0.7)
ax5.text(BEST_LAYER + 0.3, 0.02, f'Best probe\nL{BEST_LAYER}',
         color='#aaaaaa', fontsize=7.5, va='bottom')

ax5.set_ylim(0, 1.05)
ax5.legend(loc='center right', facecolor='#1a1d27',
           edgecolor=SPINE_COL, labelcolor='#cccccc', fontsize=8)


# ── Panel 6: Per-head emotional attention ─────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 1], **AX_STYLE)
style_ax(ax6,
         title='Panel 6 — Per-Head Mean Emotional Attention',
         xlabel='Attention Head', ylabel='Mean Emotional Attention')

head_ids = np.arange(len(head_emo))
bar_colors = ['#E63946' if i in att['summary']['head_summary']['top_5_emo_heads']
              else '#3a3d60'
              for i in head_ids]
ax6.bar(head_ids, head_emo, color=bar_colors, alpha=0.85)

# annotate top-5 and bottom-5
top5    = att['summary']['head_summary']['top_5_emo_heads']
bottom5 = att['summary']['head_summary']['bottom_5_emo_heads']
for h in top5:
    ax6.text(h, head_emo[h] + 0.0005, f'H{h}',
             ha='center', va='bottom', fontsize=7.5, color='#E63946')
for h in bottom5:
    ax6.text(h, head_emo[h] + 0.0005, f'H{h}',
             ha='center', va='bottom', fontsize=7.5, color='#888888')

from matplotlib.patches import Patch as _Patch
ax6.legend(
    handles=[_Patch(facecolor='#E63946', label='Top-5 heads'),
             _Patch(facecolor='#3a3d60', label='Other heads')],
    loc='lower right', facecolor='#1a1d27',
    edgecolor=SPINE_COL, labelcolor='#cccccc', fontsize=8,
)

# %%
# ── Super-title and save ───────────────────────────────────────────────────────
fig.suptitle(
    'Probe Weight × Attention Analysis — Llama-3.2-3B-Instruct (EPR cubic α=0.0001)',
    fontsize=14, fontweight='bold', color='#ffffff', y=0.97,
)

plt.savefig(OUT_FILE, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print(f'Saved: {OUT_FILE}')
plt.show()
