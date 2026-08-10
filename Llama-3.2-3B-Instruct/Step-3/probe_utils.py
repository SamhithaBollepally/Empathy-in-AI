# %%
"""Shared inference utilities for the emotion probes trained in Step-2.

Loading a saved probe bundle (emotion_probes.pt) and predicting emotions from
raw embeddings live here so Step-3 (and any other step) can reuse them without
importing — and re-running — the training script.
"""
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict


# %%
def load_probe(filepath, device, layer=None, penalty='l2'):
    """Load a probe for a given layer/penalty from a saved bundle (emotion_probes.pt).

    penalty: 'l2' (default, denser & usually more accurate) or 'l1' (sparse).
    layer=None → use the bundle's best layer for that penalty (best by prompt accuracy).
    Returns (probe: nn.Linear, scaler, classes, layer).
    """
    if penalty not in ('l1', 'l2'):
        raise ValueError(f"penalty must be 'l1' or 'l2', got {penalty!r}")
    bundle = torch.load(filepath, map_location=device, weights_only=False)
    layer  = bundle[f'best_layer_{penalty}'] if layer is None else int(layer)
    entry  = bundle['layers'][layer]

    hidden_dim  = bundle['hidden_dim']
    num_classes = len(bundle['classes'])
    probe = nn.Linear(hidden_dim, num_classes).to(device)
    probe.load_state_dict(entry[f'{penalty}_state_dict'])
    probe.eval()
    return probe, entry['scaler'], bundle['classes'], layer


# %%
def predict_emotions(embeddings, probe, scaler, classes, device):
    """Predict emotion labels from raw embeddings for the probe's layer.

    embeddings: (num_samples, hidden_dim) at the SAME layer the probe was trained on.
    Applies the saved StandardScaler, then argmax over the linear probe.
    Returns (labels: list[str], probs: np.ndarray of shape (num_samples, num_classes)).
    """
    X = scaler.transform(np.asarray(embeddings, dtype=np.float32))
    X = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = probe(X)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        idx    = logits.argmax(dim=1).cpu().numpy()
    labels = [str(classes[i]) for i in idx]
    return labels, probs


# %%
def load_response_embeddings_npz(filepath):
    """Load an 'emotion_idx_layer' keyed npz → {emotion: {idx: {layer: emb}}}.

    Mirrors the layout written by L3bI-ResponseEmbeddings.py. Empty responses
    were skipped at embedding time, so they will be absent from this dict.
    """
    data = np.load(filepath)
    embs = defaultdict(lambda: defaultdict(dict))
    for key in data.files:
        emotion, idx, layer = key.rsplit('_', 2)
        embs[emotion][int(idx)][int(layer)] = data[key]
    return embs
