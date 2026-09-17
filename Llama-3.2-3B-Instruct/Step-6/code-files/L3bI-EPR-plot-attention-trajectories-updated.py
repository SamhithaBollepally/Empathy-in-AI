# %%
# ============================================================================
# Plotting helpers for attention-flow arrays
#
# Reads:
#   attention_flow_arrays.npz
#   attention_flow_summary.json
#
# Produces example figures under Step-6/data/figures/:
#   1. Per-sample attention trajectory (one sample, selected layers/heads/bucket)
#   2. Per-emotion average attention trajectory (bucket over generation steps)
#   3. Heatmap of mean emotional attention per (tracked layer, head)
#   4. Mean attention buckets over generation steps, all layers head-averaged
#
# No model is needed; this is fast and runs on the saved .npz.
# ============================================================================

# %%
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# A plain script has no GUI event loop, so plt.show() on the macOS backend
# raises 'SystemError: NULL object passed to Py_BuildValue' during interpreter
# teardown (harmless, but noisy). Only show when actually interactive.
_INTERACTIVE = hasattr(sys, 'ps1') or 'ipykernel' in sys.modules


def _finish(fig, save_path):
    """Save the figure, then show it only if an event loop exists."""
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    if _INTERACTIVE:
        plt.show()
    else:
        plt.close(fig)   # release the figure instead of leaking it


# %%
# ── Config ───────────────────────────────────────────────────────────────────
BASE_PATH = '/Volumes/Reading/Projects/Empathy-LLM/Llama-3.2-3B-Instruct/'
DATA_PATH = BASE_PATH + 'Step-6/data/'

FLOW_FILE    = DATA_PATH + 'attention_flow_arrays.npz'
SUMMARY_FILE = DATA_PATH + 'attention_flow_summary.json'
FIG_DIR      = DATA_PATH + 'figures/'
os.makedirs(FIG_DIR, exist_ok=True)

# %%
# ── Load data ────────────────────────────────────────────────────────────────
flow = np.load(FLOW_FILE, allow_pickle=True)
with open(SUMMARY_FILE) as f:
    summary = json.load(f)['summary']

head_bucket  = flow['head_bucket']       # (T, n_track, n_heads, 5)
layer_bucket = flow['layer_bucket']      # (T, 28, 5)
step_sample  = flow['step_sample']       # (T,)
step_num     = flow['step_num']          # (T,)
sample_emotion = flow['sample_emotion']  # (n_samples,)

bucket_order  = [str(b) for b in flow['bucket_order']]
track_layers  = [int(l) for l in flow['track_layers']]
layer_ids     = [int(l) for l in flow['layer_ids']]
BUCKET_IDX    = {b: i for i, b in enumerate(bucket_order)}
TRACK_LAYER_IDX = {L: i for i, L in enumerate(track_layers)}

print(f'Loaded: head_bucket {head_bucket.shape}, layer_bucket {layer_bucket.shape}')
print(f'Tracked layers: {track_layers}')
print(f'Bucket order: {bucket_order}')

# %%
# ── Helper: reconstruct per-sample slices ────────────────────────────────────
def sample_indices(sample_id):
    """Return row indices in the flat step arrays belonging to one sample."""
    return np.where(step_sample == sample_id)[0]


def emotion_sample_ids(emotion):
    """Return all sample ids for a given emotion."""
    return np.where(sample_emotion == emotion)[0]


def _check_head_layer(layer, head):
    """Validate a (layer, head) pair against what was actually stored."""
    if layer not in TRACK_LAYER_IDX:
        raise ValueError(
            f'layer {layer} has no per-head data. Head-level detail was only '
            f'stored for TRACK_LAYERS={track_layers}. Use head=None to plot the '
            f'head-averaged value, available for all layers {layer_ids[0]}..{layer_ids[-1]}.')
    n_heads = head_bucket.shape[2]
    if not 0 <= head < n_heads:
        raise ValueError(f'head {head} out of range (0..{n_heads - 1})')


def _check_layer(layer):
    if layer not in layer_ids:
        raise ValueError(f'layer {layer} out of range '
                         f'({layer_ids[0]}..{layer_ids[-1]}, hidden-state numbering)')


def _check_bucket(bucket):
    if bucket not in BUCKET_IDX:
        raise ValueError(f'unknown bucket {bucket!r}; choose from {bucket_order}')


# %%
# ── Plot 1: per-sample trajectory, one (layer, head) and bucket ────────────────
def plot_sample_trajectory(sample_id, layer, head, bucket='emotional',
                           title=None, save_path=None, ax=None):
    """Plot attention to `bucket` for one sample across generation steps."""
    _check_head_layer(layer, head)
    _check_bucket(bucket)
    rows = sample_indices(sample_id)
    if len(rows) == 0:
        raise ValueError(f'sample {sample_id} not found')

    li = TRACK_LAYER_IDX[layer]
    bi = BUCKET_IDX[bucket]
    y = head_bucket[rows, li, head, bi]
    x = step_num[rows]

    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    ax.plot(x, y, marker='o', markersize=3)
    ax.set_xlabel('generation step')
    ax.set_ylabel(f'{bucket} attention')
    emotion = str(sample_emotion[sample_id])
    ax.set_title(title or f'sample {sample_id} | {emotion} | layer {layer} head {head}')
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)

    if owns_fig:
        _finish(fig, save_path)
    return ax


# %%
# ── Plot 2: per-emotion average trajectory for a bucket ──────────────────────
def plot_emotion_trajectory(emotion, layer=None, head=None, bucket='emotional',
                            save_path=None):
    """Average trajectory across all samples of one emotion.

    If head is None, use the head-averaged `layer_bucket`, available for every
    layer. If head is given, use `head_bucket`, available only for TRACK_LAYERS.
    """
    if layer is None:
        raise ValueError('layer is required (hidden-state numbering, 1..28)')
    _check_bucket(bucket)
    if head is None:
        _check_layer(layer)
    else:
        _check_head_layer(layer, head)

    sids = emotion_sample_ids(emotion)
    if len(sids) == 0:
        raise ValueError(
            f'no samples for emotion {emotion!r}; '
            f'available: {sorted(set(map(str, sample_emotion)))}')

    bi = BUCKET_IDX[bucket]

    # collect each sample's trajectory, padded with nan so lengths can differ
    traces = []
    for sid in sids:
        rows = sample_indices(sid)
        if head is None:
            li = layer_ids.index(layer)   # layer is in 1..28
            vals = layer_bucket[rows, li, bi]
        else:
            li = TRACK_LAYER_IDX[layer]
            vals = head_bucket[rows, li, head, bi]
        traces.append(vals)

    max_len = max(len(t) for t in traces)
    mat = np.full((len(traces), max_len), np.nan)
    for i, t in enumerate(traces):
        mat[i, :len(t)] = t

    mean_traj = np.nanmean(mat, axis=0)
    std_traj  = np.nanstd(mat, axis=0, ddof=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(mean_traj))
    ax.plot(x, mean_traj, color='C0')
    ax.fill_between(x, mean_traj - std_traj, mean_traj + std_traj,
                    color='C0', alpha=0.2)
    ax.set_xlabel('generation step')
    ax.set_ylabel(f'{bucket} attention')
    head_label = f'head {head}' if head is not None else 'head-averaged'
    ax.set_title(f'{emotion} | layer {layer} | {head_label} | mean ± std')
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)

    _finish(fig, save_path)
    return ax


# %%
# ── Plot 3: heatmap of mean emotional attention per (tracked layer, head) ──
def plot_head_heatmap(bucket='emotional', save_path=None):
    """Mean attention to `bucket` for every (tracked layer, head) pair.

    Averaged over all generation steps pooled together, so longer responses
    contribute more steps - the same weighting the summary JSON uses.
    """
    _check_bucket(bucket)
    bi = BUCKET_IDX[bucket]
    mat = head_bucket[:, :, :, bi].mean(axis=0)   # (n_track, n_heads)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(mat, aspect='auto', cmap='viridis')
    ax.set_yticks(np.arange(len(track_layers)))
    ax.set_yticklabels(track_layers)
    ax.set_xticks(np.arange(head_bucket.shape[2]))
    ax.set_xlabel('head')
    ax.set_ylabel('layer')
    ax.set_title(f'mean {bucket} attention per (tracked layer, head)')
    plt.colorbar(im, ax=ax, label=f'{bucket} attention')

    # Annotate top 10 cells from the summary
    top = summary.get('top_emotional_heads', [])
    for h in top:
        if h['layer'] in track_layers:
            yi = track_layers.index(h['layer'])
            xi = h['head']
            ax.text(xi, yi, f"{h['mean_emotional_attention']:.3f}",
                    ha='center', va='center', color='white', fontsize=6)

    _finish(fig, save_path)
    return ax


# %%
# ── Plot 4: mean bucket attention over steps, all layers head-averaged ───────
def plot_layer_bucket_over_steps(layer, save_path=None):
    """For a single layer, plot the mean attention to each bucket over time."""
    _check_layer(layer)
    li = layer_ids.index(layer)   # layer is 1..28
    sids = np.arange(int(step_sample.max()) + 1)

    # collect per-step bucket vectors per sample, then average across samples
    max_len = int(step_num.max()) + 1
    mats = {b: np.full((len(sids), max_len), np.nan)
            for b in bucket_order}

    for i, sid in enumerate(sids):
        rows = sample_indices(sid)
        for b in bucket_order:
            bi = BUCKET_IDX[b]
            vals = layer_bucket[rows, li, bi]
            mats[b][i, :len(vals)] = vals

    fig, ax = plt.subplots(figsize=(10, 5))
    for b in bucket_order:
        mean_traj = np.nanmean(mats[b], axis=0)
        ax.plot(mean_traj, label=b)
    ax.set_xlabel('generation step')
    ax.set_ylabel('mean attention')
    ax.set_title(f'layer {layer} | mean bucket attention over generation steps')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    _finish(fig, save_path)
    return ax


# %%
# ── Example run ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # 1. Trajectory of the very first sample at the strongest reported layer/head.
    top = summary.get('top_emotional_heads') or []
    top_head = top[0] if top else {}
    example_layer = int(top_head.get('layer', track_layers[0]))
    example_head  = int(top_head.get('head', 0))
    print(f'Example (layer, head) = ({example_layer}, {example_head})')
    plot_sample_trajectory(
        sample_id=0, layer=example_layer, head=example_head, bucket='emotional',
        save_path=os.path.join(FIG_DIR, 'example_sample_trajectory.png'))

    # 2. Per-emotion average for the same (layer, head) — choose the first emotion.
    plot_emotion_trajectory(
        emotion=str(sample_emotion[0]), layer=example_layer, head=example_head,
        bucket='emotional',
        save_path=os.path.join(FIG_DIR, 'example_emotion_trajectory.png'))

    # 3. Heatmap of emotional attention across all tracked (layer, head) pairs.
    plot_head_heatmap(
        bucket='emotional',
        save_path=os.path.join(FIG_DIR, 'example_head_heatmap.png'))

    # 4. Bucket dynamics at the probe layer across generation steps.
    plot_layer_bucket_over_steps(
        layer=int(summary['config']['probe_layer']),
        save_path=os.path.join(FIG_DIR, 'example_probe_layer_buckets.png'))
