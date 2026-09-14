# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %%
# ============================================================================
# Probe analysis of steered responses (best-layer readout) + COHERENCE metric.
#
# Same as probe-analysis.py (L1 @ best_layer_l1, L2 @ best_layer_l2, masked-mean
# embeddings, target accuracy vs unsteered baseline) but ALSO measures a
# repetition/degeneration score per response, so a high target accuracy that is
# really just the model looping emotion words ("grateful grateful grateful") is
# flagged rather than counted as success.
#
# Points at the FULL contrastive run: Outputs/SA_full/.
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
PROBE_PATH = STEP2_PATH + 'emotion_probes.pt'

# Full contrastive run inputs/outputs.
STEER_DIR    = BASE_PATH + 'Outputs/SA_full/'
OUT_DIR      = STEP4_PATH + 'probe_eval_SA_full/'
SUMMARY_PATH = STEP4_PATH + 'steering_probe_eval_SA_full.json'
os.makedirs(OUT_DIR, exist_ok=True)

NULL_LABEL = 'Null'
BATCH_SIZE = 8
MAX_LENGTH = 512
BASELINE_N_PER_EMOTION = None   # None -> full 200/emotion (match the full steered run)
REP_N = 3                       # n-gram size for the repetition score
DEGENERATE_THRESHOLD = 0.3      # repetition >= this counts as degenerate

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
    """Return (probe: nn.Linear, scaler, classes, layer) for 'l1' or 'l2'."""
    layer = bundle[f'best_layer_{penalty}']
    entry = bundle['layers'][layer]
    probe = nn.Linear(bundle['hidden_dim'], len(bundle['classes'])).to(device)
    probe.load_state_dict(entry[f'{penalty}_state_dict'])
    probe.eval()
    return probe, entry['scaler'], [str(c) for c in bundle['classes']], layer


def predict_labels(emb, probe, scaler, classes):
    X = scaler.transform(np.asarray(emb, dtype=np.float32))
    X = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        idx = probe(X).argmax(dim=1).cpu().numpy()
    return np.array([classes[i] for i in idx])

# %%
def repetition_score(text, n=REP_N):
    """Fraction of repeated word n-grams: 0 = all unique, ->1 = heavy looping.

    Degenerate steering ("grateful grateful grateful") scores high; normal text low.
    Falls back to smaller n for very short responses.
    """
    toks = text.split()
    while n > 1 and len(toks) < n + 1:
        n -= 1
    if len(toks) < n + 1:
        return 0.0
    ngrams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)

# %%
def masked_mean_pool(hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
    summed = (hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


class _StopForward(Exception):
    """Raised inside a hook to abort the forward pass once we have what we need."""


def embed_texts(texts, layers):
    """Embed texts -> {layer: np.ndarray(n, hidden_dim)} (masked mean pool).

    Early-stops after the deepest needed layer and skips lm_head (runs model.model).
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
                    model.model(**inputs)
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
bundle = torch.load(PROBE_PATH, map_location=device, weights_only=False)
probe_l1, scaler_l1, classes, layer_l1 = load_probe(bundle, device, 'l1')
probe_l2, scaler_l2, _,       layer_l2 = load_probe(bundle, device, 'l2')
NEED_LAYERS = sorted({layer_l1, layer_l2})
print(f"L1 probe @ layer {layer_l1} | L2 probe @ layer {layer_l2}")

# %%
def evaluate_group(data):
    """Predict L1/L2 emotion + repetition per response. Congruent target == emotion key."""
    records = {e: [] for e in data}
    texts, refs = [], []
    n_null = 0
    for emotion, items in data.items():
        for idx, it in enumerate(items):
            text = (it.get('response', '') or '').strip()
            rec = {'index': idx, 'true_emotion': emotion, 'pred_l1': NULL_LABEL,
                   'pred_l2': NULL_LABEL, 'repetition': None}
            if text:
                rec['repetition'] = round(repetition_score(text), 4)
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
    reps = [r['repetition'] for r in pred if r['repetition'] is not None]
    mean_rep = round(float(np.mean(reps)), 4) if reps else None
    degen = round(float(np.mean([r >= DEGENERATE_THRESHOLD for r in reps])), 4) if reps else None
    # "Clean" accuracy: only over coherent (non-degenerate) responses.
    clean = [r for r in pred if (r['repetition'] or 0) < DEGENERATE_THRESHOLD]
    clean_acc_l2 = round(float(np.mean([r['pred_l2'] == r['true_emotion'] for r in clean])), 4) if clean else None
    metrics = {'n_total': len(flat), 'n_predicted': len(pred), 'n_null': n_null,
               'l1_layer': int(layer_l1), 'l2_layer': int(layer_l2),
               'target_acc_l1': acc('pred_l1'), 'target_acc_l2': acc('pred_l2'),
               'mean_repetition': mean_rep, 'degenerate_frac': degen,
               'clean_acc_l2': clean_acc_l2, 'n_clean': len(clean)}
    return records, metrics

# %%
# ---- Baseline: unsteered utterances (coeff = 0), matched N -------------------
summary = []
with open(BASE_PATH + 'Outputs/zeroshot_responses.json') as f:
    zs = json.load(f)
baseline_data = {e: (zs[e] if BASELINE_N_PER_EMOTION is None else zs[e][:BASELINE_N_PER_EMOTION])
                 for e in EMOTIONS}
print("\n=== BASELINE (unsteered) ===")
base_records, base_metrics = evaluate_group(baseline_data)
with open(OUT_DIR + 'baseline_predictions.json', 'w') as f:
    json.dump({'metrics': base_metrics, 'predictions': base_records}, f, indent=2)
summary.append({'output_file': 'baseline', 'layer_steered': None, 'coeff': 0.0, **base_metrics})
BASE_L1, BASE_L2 = base_metrics['target_acc_l1'], base_metrics['target_acc_l2']
print(f"  L1={BASE_L1} | L2={BASE_L2} | mean_rep={base_metrics['mean_repetition']} "
      f"| degenerate={base_metrics['degenerate_frac']}")

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
    print(f"  L2={metrics['target_acc_l2']} | clean_L2={metrics['clean_acc_l2']} "
          f"| mean_rep={metrics['mean_repetition']} | degenerate={metrics['degenerate_frac']} "
          f"| null={metrics['n_null']}")

# %%
# ---- Report: rank by CLEAN L2 accuracy (accuracy on coherent responses) ------
print(f"\nBaseline: L1@{layer_l1}={BASE_L1}  L2@{layer_l2}={BASE_L2}  "
      f"(rep={base_metrics['mean_repetition']})")
print("\nConfigs by clean L2 accuracy (coherent responses only):")
rows = [r for r in summary if r['output_file'] != 'baseline']
hdr = f"  {'layer':>5} {'coeff':>5} | {'L2':>6} {'cleanL2':>7} {'rep':>6} {'degen%':>7} {'Δclean':>7}"
print(hdr)
for r in sorted(rows, key=lambda r: (r['clean_acc_l2'] or 0), reverse=True):
    d = (r['clean_acc_l2'] - BASE_L2) if (r['clean_acc_l2'] is not None and BASE_L2 is not None) else None
    dt = f"{d:+.3f}" if d is not None else "  n/a"
    print(f"  {str(r['layer_steered']):>5} {str(r['coeff']):>5} | "
          f"{r['target_acc_l2']:>6} {r['clean_acc_l2']:>7} {r['mean_repetition']:>6} "
          f"{r['degenerate_frac']:>7} {dt:>7}")

print(f"\nDone. Summary: {SUMMARY_PATH}")
