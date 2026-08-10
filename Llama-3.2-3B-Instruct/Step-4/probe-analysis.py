# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %%
# ============================================================================
# Probe analysis of steered responses (best-layer readout, mirrors Step-3).
#
# For every steering config in Outputs/Steering-initial/, embed the generated
# responses (same masked-mean pooling as Step-3/L3bI-ResponseEmbeddings.py) and
# predict their emotion with the L1 and L2 probes at their BEST layer only
# (exactly as Step-3/Analysis.py did): L1 -> best_layer_l1, L2 -> best_layer_l2.
#
# Congruent steering means the TARGET emotion == the emotion the response was
# grouped under, so "target accuracy" = fraction predicted == that emotion.
# Compared against the same-utterance UNSTEERED baseline (coeff = 0).
# ============================================================================
import os
import json
import glob
import getpass
import numpy as np
import torch
import torch.nn as nn
import huggingface_hub
from transformers import AutoTokenizer, AutoModelForCausalLM

# %%
BASE_PATH = '/Volumes/Reading/Projects/Empathy-LLM/Llama-3.2-3B-Instruct/'
#BASE_PATH = '/content/drive/MyDrive/Project-4/'

EMOTIONS = ['afraid', 'angry', 'anxious', 'devastated', 'lonely', 'sad',
            'terrified', 'furious', 'grateful', 'hopeful', 'faithful']

STEP2_PATH = BASE_PATH + 'Step-2/'
STEP4_PATH = BASE_PATH + 'Step-4/'
STEER_DIR  = BASE_PATH + 'Outputs/Steering-initial/'
PROBE_PATH = STEP2_PATH + 'emotion_probes.pt'

OUT_DIR      = STEP4_PATH + 'probe_eval/'          # per-config predictions
SUMMARY_PATH = STEP4_PATH + 'steering_probe_eval.json'
os.makedirs(OUT_DIR, exist_ok=True)

NULL_LABEL = 'Null'
BATCH_SIZE = 8
MAX_LENGTH = 512
BASELINE_N_PER_EMOTION = 20   # unsteered baseline over the same first-N utterances

# %%
token = getpass.getpass("Enter your Hugging Face token: ")
huggingface_hub.login(token=token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# On CPU: use all cores and float32 (fp16 is unsupported/slow on CPU).
ON_CPU = device.type == "cpu"
if ON_CPU:
    torch.set_num_threads(os.cpu_count() or 1)
    print(f"CPU threads: {torch.get_num_threads()}")

# %%
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32 if ON_CPU else torch.float16,
    low_cpu_mem_usage=True,
    device_map=None if ON_CPU else "auto",
)
model.eval()
if ON_CPU:
    model.to(device)
MODEL_DEVICE = next(model.parameters()).device
print(f"Model loaded on {MODEL_DEVICE} ({next(model.parameters()).dtype}).")

# %%
# ---- Probe inference helpers (inlined from the former probe_utils.py) --------
def load_probe(bundle, device, penalty):
    """Return (probe: nn.Linear, scaler, classes, layer) for 'l1' or 'l2'.

    Uses the bundle's best layer for that penalty (matches Step-3 layer=None).
    """
    layer = bundle[f'best_layer_{penalty}']
    entry = bundle['layers'][layer]
    probe = nn.Linear(bundle['hidden_dim'], len(bundle['classes'])).to(device)
    probe.load_state_dict(entry[f'{penalty}_state_dict'])
    probe.eval()
    return probe, entry['scaler'], [str(c) for c in bundle['classes']], layer


def predict_labels(emb, probe, scaler, classes):
    """Predicted emotion labels for embeddings (n, hidden_dim)."""
    X = scaler.transform(np.asarray(emb, dtype=np.float32))
    X = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        idx = probe(X).argmax(dim=1).cpu().numpy()
    return np.array([classes[i] for i in idx])

# %%
def masked_mean_pool(hidden_state, attention_mask):
    """Mean-pool token embeddings using the attention mask (ignores padding)."""
    mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
    summed = (hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


class _StopForward(Exception):
    """Raised inside a hook to abort the forward pass once we have what we need."""


def embed_texts(texts, layers):
    """Embed texts -> {layer: np.ndarray(n, hidden_dim)} (masked mean pool).

    Speed: we only need a few readout layers, so we capture their raw block
    outputs via hooks and early-stop after the deepest one — skipping the upper
    decoder layers AND the large lm_head. Runs on the base model (model.model),
    so no vocab projection is computed. Captured tensors equal hidden_states[L].
    """
    stop_layer = max(layers)
    captured = {}

    def make_hook(L):
        def hook(module, inputs, output):
            captured[L] = output[0] if isinstance(output, tuple) else output
            if L == stop_layer:
                raise _StopForward
        return hook

    handles = [model.model.layers[L - 1].register_forward_hook(make_hook(L)) for L in layers]
    out = {L: [] for L in layers}
    try:
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            inputs = tokenizer(batch, return_tensors='pt', padding=True,
                               truncation=True, max_length=MAX_LENGTH).to(MODEL_DEVICE)
            captured.clear()
            with torch.inference_mode():
                try:
                    model.model(**inputs)          # base model only (no lm_head)
                except _StopForward:
                    pass
            attn = inputs['attention_mask']
            for L in layers:
                pooled = masked_mean_pool(captured[L], attn).float().cpu().numpy()
                out[L].append(pooled)
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        for h in handles:
            h.remove()
    return {L: np.concatenate(v, axis=0) for L, v in out.items()}

# %%
# Load the L1 and L2 probes at their best layer only (mirrors Step-3).
bundle = torch.load(PROBE_PATH, map_location=device, weights_only=False)
probe_l1, scaler_l1, classes, layer_l1 = load_probe(bundle, device, 'l1')
probe_l2, scaler_l2, _,       layer_l2 = load_probe(bundle, device, 'l2')
NEED_LAYERS = sorted({layer_l1, layer_l2})
print(f"L1 probe @ layer {layer_l1} | L2 probe @ layer {layer_l2}")

# %%
def evaluate_group(data):
    """data: {emotion: [ {response, ...} ]}. Congruent target == emotion key.

    Predicts with L1 (layer_l1) and L2 (layer_l2). Empty responses -> NULL_LABEL,
    excluded from accuracy. Returns (records_by_emotion, metrics).
    """
    records = {e: [] for e in data}
    texts, refs = [], []   # refs: (emotion, position)
    n_null = 0
    for emotion, items in data.items():
        for idx, it in enumerate(items):
            text = (it.get('response', '') or '').strip()
            rec = {'index': idx, 'true_emotion': emotion,
                   'pred_l1': NULL_LABEL, 'pred_l2': NULL_LABEL}
            if text:
                refs.append((emotion, len(records[emotion])))
                texts.append(text)
            else:
                n_null += 1
            records[emotion].append(rec)

    if texts:
        embs = embed_texts(texts, NEED_LAYERS)
        lab1 = predict_labels(embs[layer_l1], probe_l1, scaler_l1, classes)
        lab2 = predict_labels(embs[layer_l2], probe_l2, scaler_l2, classes)
        for (emotion, pos), a, b in zip(refs, lab1, lab2):
            records[emotion][pos]['pred_l1'] = str(a)
            records[emotion][pos]['pred_l2'] = str(b)

    flat = [r for rs in records.values() for r in rs]
    pred = [r for r in flat if r['pred_l2'] != NULL_LABEL]
    def acc(k):
        return round(float(np.mean([r[k] == r['true_emotion'] for r in pred])), 4) if pred else None
    metrics = {'n_total': len(flat), 'n_predicted': len(pred), 'n_null': n_null,
               'l1_layer': int(layer_l1), 'l2_layer': int(layer_l2),
               'target_acc_l1': acc('pred_l1'), 'target_acc_l2': acc('pred_l2')}
    return records, metrics

# %%
# ---- Baseline: same first-N unsteered utterances (coeff = 0) -----------------
summary = []
with open(BASE_PATH + 'Outputs/zeroshot_responses.json') as f:
    zs = json.load(f)
baseline_data = {e: zs[e][:BASELINE_N_PER_EMOTION] for e in EMOTIONS}
print(f"\n=== BASELINE (unsteered, first {BASELINE_N_PER_EMOTION}/emotion) ===")
base_records, base_metrics = evaluate_group(baseline_data)
with open(OUT_DIR + 'baseline_predictions.json', 'w') as f:
    json.dump({'metrics': base_metrics, 'predictions': base_records}, f, indent=2)
summary.append({'output_file': 'baseline', 'layer_steered': None, 'coeff': 0.0, **base_metrics})
print(f"  L1 target_acc={base_metrics['target_acc_l1']} | "
      f"L2 target_acc={base_metrics['target_acc_l2']}")
BASE_L1, BASE_L2 = base_metrics['target_acc_l1'], base_metrics['target_acc_l2']

# %%
# ---- Every steered config ----------------------------------------------------
files = sorted(glob.glob(STEER_DIR + 'steered_*.json'))
print(f"\nFound {len(files)} steered config files.")

for path in files:
    name = os.path.basename(path)
    with open(path) as f:
        data = json.load(f)
    first = next(it for items in data.values() for it in items)
    layer_steered, coeff = first.get('steer_layer'), first.get('coeff')

    print(f"\n=== {name} (steer layer {layer_steered}, coeff {coeff}) ===")
    records, metrics = evaluate_group(data)
    with open(OUT_DIR + name, 'w') as f:
        json.dump({'layer_steered': layer_steered, 'coeff': coeff,
                   'metrics': metrics, 'predictions': records}, f, indent=2)
    summary.append({'output_file': name, 'layer_steered': layer_steered,
                    'coeff': coeff, **metrics})
    with open(SUMMARY_PATH, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  L1 target_acc={metrics['target_acc_l1']} | "
          f"L2 target_acc={metrics['target_acc_l2']} | null={metrics['n_null']}")

# %%
# ---- Report: top configs by L2 target accuracy vs baseline -------------------
print(f"\nBaseline: L1@{layer_l1}={BASE_L1}  L2@{layer_l2}={BASE_L2}")
print("\nTop steered configs by L2 target accuracy:")
rows = [r for r in summary if r['output_file'] != 'baseline']
for r in sorted(rows, key=lambda r: (r['target_acc_l2'] or 0), reverse=True)[:12]:
    d1 = (r['target_acc_l1'] - BASE_L1) if (r['target_acc_l1'] is not None and BASE_L1 is not None) else None
    d2 = (r['target_acc_l2'] - BASE_L2) if (r['target_acc_l2'] is not None and BASE_L2 is not None) else None
    d1t = f"{d1:+.3f}" if d1 is not None else "n/a"
    d2t = f"{d2:+.3f}" if d2 is not None else "n/a"
    print(f"  steer L{r['layer_steered']:>2} c{r['coeff']:<4} | "
          f"L1={r['target_acc_l1']} (Δ{d1t}) | L2={r['target_acc_l2']} (Δ{d2t})")

print(f"\nDone. Summary: {SUMMARY_PATH}")
