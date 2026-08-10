# %%
import os
import sys
import json
import numpy as np
import torch

# %%
BASE_PATH = '/Volumes/Reading/Projects/Empathy-LLM/Llama-3.2-3B-Instruct/'
#BASE_PATH = '/content/drive/MyDrive/Project-4/'

# Reuse the shared probe inference helpers (no training re-run).
sys.path.insert(0, BASE_PATH)
from probe_utils import load_probe, predict_emotions, load_response_embeddings_npz

OUTPUTS_PATH = BASE_PATH + 'Outputs/'
STEP2_PATH   = BASE_PATH + 'Step-2/'
STEP3_PATH   = BASE_PATH + 'Step-3/'
os.makedirs(STEP3_PATH, exist_ok=True)

PROBE_PATH = STEP2_PATH + 'emotion_probes.pt'
NULL_LABEL = 'Null'      # emitted when the response cell is empty
PENALTIES  = ['l1', 'l2']  # run and save predictions for both probe types

# (responses json, response embeddings npz, output filename prefix)
RESPONSE_SETS = [
    ('zeroshot_responses.json', 'zeroshot_response_embeddings.npz', 'zeroshot'),
    ('CoT_responses.json',      'CoT_response_embeddings.npz',      'CoT'),
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# %%
def predict_response_emotions(responses, embs, probe, scaler, classes, layer, device):
    """Predict an emotion for every response, preserving source order.

    Empty response cells → NULL_LABEL (no embedding was produced for them).
    Non-empty responses are batched through the probe in a single pass.

    Returns (records_by_emotion, n_null, n_missing) where each record is
    {index, true_emotion, predicted_emotion, confidence}.
    """
    records = {emotion: [] for emotion in responses}
    to_predict_vecs, to_predict_refs = [], []  # refs: (emotion, position in records list)
    n_null = n_missing = 0

    for emotion, items in responses.items():
        for idx, item in enumerate(items):
            text = (item.get('response', '') or '').strip()
            rec = {'index': idx, 'true_emotion': emotion,
                   'predicted_emotion': NULL_LABEL, 'confidence': None}

            if not text:
                n_null += 1
            else:
                vec = embs.get(emotion, {}).get(idx, {}).get(layer)
                if vec is None:
                    # Text present but no embedding (e.g. dropped upstream) — leave as Null.
                    n_missing += 1
                else:
                    to_predict_refs.append((emotion, len(records[emotion])))
                    to_predict_vecs.append(vec)
            records[emotion].append(rec)

    if to_predict_vecs:
        labels, probs = predict_emotions(np.stack(to_predict_vecs), probe, scaler, classes, device)
        for (emotion, pos), label, prob in zip(to_predict_refs, labels, probs):
            records[emotion][pos]['predicted_emotion'] = label
            records[emotion][pos]['confidence'] = round(float(prob.max()), 4)

    return records, n_null, n_missing

# %%
# Load response data + embeddings once, then predict with each probe (L1 and L2).
for resp_name, emb_name, prefix in RESPONSE_SETS:
    print(f'\n=== {resp_name} ===')
    with open(OUTPUTS_PATH + resp_name) as f:
        responses = json.load(f)
    embs = load_response_embeddings_npz(OUTPUTS_PATH + emb_name)

    for penalty in PENALTIES:
        # layer=None → best layer for this penalty as chosen during Step-2 probing.
        probe, scaler, classes, layer = load_probe(PROBE_PATH, device, layer=None, penalty=penalty)

        records, n_null, n_missing = predict_response_emotions(
            responses, embs, probe, scaler, classes, layer, device)

        # Accuracy over predicted (non-Null) responses: does the probe recover the target emotion?
        predicted = [r for recs in records.values() for r in recs
                     if r['predicted_emotion'] != NULL_LABEL]
        n_correct = sum(r['predicted_emotion'] == r['true_emotion'] for r in predicted)
        total = sum(len(recs) for recs in records.values())
        acc = n_correct / len(predicted) if predicted else 0.0

        print(f'  [{penalty.upper()} | layer {layer}] predicted: {len(predicted)}/{total} | '
              f'null: {n_null} | missing: {n_missing} | accuracy: {acc:.4f}')

        out = {
            'penalty': penalty,
            'probe_layer': int(layer),
            'classes': [str(c) for c in classes],
            'summary': {
                'total': total, 'predicted': len(predicted),
                'null_empty': n_null, 'missing_embedding': n_missing,
                'accuracy': round(acc, 4),
            },
            'predictions': records,
        }
        out_name = f'{prefix}_emotion_predictions_{penalty}.json'
        with open(STEP3_PATH + out_name, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'    Saved: {out_name}')

print('\nDone.')
