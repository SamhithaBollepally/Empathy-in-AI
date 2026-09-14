# %%
# ============================================================================
# Step 5b — Probe Target Accuracy Evaluation
#
# Evaluates EPR-generated responses at every layer using the trained probes.
# For each generated response, extracts hidden states at all 29 layers,
# applies the corresponding L2 probe, and records whether the top-1
# predicted emotion matches the target emotion.
#
# Outputs:
#   - per_layer_accuracy_{filename}.json     (overall + per-emotion accuracy)
#   - per_response_predictions_{filename}.npz (optional, for error analysis)
# ============================================================================

# %%
import os
import gc
import json
import getpass
import numpy as np
import torch
import torch.nn as nn
import huggingface_hub
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

# %%
# ── Config ───────────────────────────────────────────────────────────────────
BASE_PATH = '/content/drive/MyDrive/Project-4/'
# BASE_PATH = '/Volumes/Reading/Projects/Empathy-LLM/Llama-3.2-3B-Instruct/'

STEP2_PATH = BASE_PATH + 'Step-2/'
STEP5_PATH = BASE_PATH + 'Step-5/'

# Path to the EPR responses file (change as needed)
EPR_RESULT_FILE = STEP5_PATH + 'Steering_distributionepr_photocopy_c0.2_pca50.json'
# Or use: EPR_RESULT_FILE = STEP5_PATH + 'epr_photocopy_c0.2_pca50.json'

BATCH_SIZE = 32     # A100 can handle this for short responses

# %%
EMOTIONS = ['afraid', 'angry', 'anxious', 'devastated', 'lonely', 'sad',
            'terrified', 'furious', 'grateful', 'hopeful', 'faithful']

# %%
# ── Load EPR results ─────────────────────────────────────────────────────────
with open(EPR_RESULT_FILE) as f:
    results = json.load(f)

print(f"Loaded responses: {sum(len(v) for v in results.values())} total")

# Flatten for processing: each item = (emotion, response_text)
flat_items = []
for emotion in EMOTIONS:
    for item in results.get(emotion, []):
        flat_items.append((emotion, item['response']))
print(f"Flattened items: {len(flat_items)}")

# %%
# ── Authentication and model load ────────────────────────────────────────────
token = getpass.getpass("HuggingFace token: ")
huggingface_hub.login(token=token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | VRAM: {props.total_memory / 1e9:.1f} GB")

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = 'right'    # right-pad to keep last token meaningful

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    output_hidden_states=True,
)
model.eval()
print("Model loaded.")

# %%
# ── Load probes and build per-layer torch tensors ────────────────────────────
bundle = torch.load(STEP2_PATH + 'emotion_probes.pt', map_location='cpu', weights_only=False)

classes = [str(c) for c in bundle['classes']]   # sorted class order
class_to_idx = {c: i for i, c in enumerate(classes)}
print(f"Probe classes: {classes}")

num_layers = len(bundle['layers'])
print(f"Probe layers: {num_layers}")

probe_weights = {}    # {layer: (11, 3072) tensor}
probe_bias    = {}    # {layer: (11,) tensor}
scaler_mean   = {}    # {layer: (3072,) tensor}
scaler_scale  = {}    # {layer: (3072,) tensor}

for L in range(num_layers):
    layer_data = bundle['layers'][L]
    scaler = layer_data['scaler']

    lin = nn.Linear(bundle['hidden_dim'], len(classes), bias=True)
    lin.load_state_dict(layer_data['l2_state_dict'])
    lin.eval()

    probe_weights[L] = lin.weight.detach().to(device).to(torch.float16)
    probe_bias[L]    = lin.bias.detach().to(device).to(torch.float16)
    scaler_mean[L]   = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    scaler_scale[L]  = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)

print("Probe tensors prepared.")

# %%
# ── Evaluation loop ──────────────────────────────────────────────────────────
# Accumulators:
#   correct[L][emotion] = number of correct top-1 predictions
#   total[L][emotion]   = number of samples evaluated
#   all_correct[L]      = total correct across all emotions at layer L

all_total        = defaultdict(int)
correct_by_layer = defaultdict(lambda: defaultdict(int))
total_by_layer   = defaultdict(lambda: defaultdict(int))

# Optional: per-response predictions for later error analysis
# keyed by (emotion, response_index)
per_response = {}

def evaluate_batch(responses, targets, start_idx):
    """
    responses : list of response strings
    targets   : list of target emotion strings
    start_idx : global start index for per_response storage
    """
    inputs = tokenizer(
        responses,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.inference_mode():
        model_out = model(**inputs, output_hidden_states=True)
    hidden_states = model_out.hidden_states   # tuple of 29 tensors

    attention_mask = inputs['attention_mask']  # (batch, seq)
    mask_sum       = attention_mask.sum(dim=1, keepdim=True).float()  # (batch, 1)
    mask_expanded  = attention_mask.unsqueeze(-1).float()             # (batch, seq, 1)

    for L, h in enumerate(hidden_states):
        # Mean-pool over non-padding tokens
        pooled = (h.float() * mask_expanded).sum(dim=1) / (mask_sum + 1e-8)   # (batch, 3072)

        # Standardise (same as sklearn scaler)
        scaled = (pooled - scaler_mean[L]) / scaler_scale[L]
        scaled = scaled.to(torch.float16)

        # Probe forward
        logits = scaled @ probe_weights[L].T + probe_bias[L]   # (batch, 11)
        preds  = logits.argmax(dim=1).cpu().numpy()            # (batch,)

        for i, target in enumerate(targets):
            tidx = class_to_idx[target]
            pred = preds[i]
            all_total[L] += 1
            total_by_layer[L][target] += 1
            if pred == tidx:
                correct_by_layer[L][target] += 1
                # Store per-response prediction for analysis
                if L == num_layers - 1:
                    global_idx = start_idx + i
                    per_response[global_idx] = {
                        'target': target,
                        'predicted': classes[pred],
                        'correct': True,
                    }
            else:
                if L == num_layers - 1:
                    global_idx = start_idx + i
                    per_response[global_idx] = {
                        'target': target,
                        'predicted': classes[pred],
                        'correct': False,
                    }

    del inputs, model_out, hidden_states
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# %%
print("\nEvaluating responses...")
for batch_start in range(0, len(flat_items), BATCH_SIZE):
    batch = flat_items[batch_start:batch_start + BATCH_SIZE]
    responses = [r for _, r in batch]
    targets   = [e for e, _ in batch]
    evaluate_batch(responses, targets, batch_start)

    if (batch_start // BATCH_SIZE) % 5 == 0:
        print(f"  processed {min(batch_start + BATCH_SIZE, len(flat_items))}/{len(flat_items)}")

print("Evaluation complete.")

# %%
# ── Compute accuracies ───────────────────────────────────────────────────────
overall_acc_by_layer = {}
for L in range(num_layers):
    total = all_total[L]
    correct = sum(correct_by_layer[L].values())
    overall_acc_by_layer[L] = round(correct / total, 4) if total else None

per_emotion_acc = {}
for L in range(num_layers):
    per_emotion_acc[L] = {}
    for emotion in EMOTIONS:
        total = total_by_layer[L].get(emotion, 0)
        correct = correct_by_layer[L].get(emotion, 0)
        per_emotion_acc[L][emotion] = round(correct / total, 4) if total > 0 else None

# %%
# ── Save results ─────────────────────────────────────────────────────────────
base_filename = os.path.splitext(os.path.basename(EPR_RESULT_FILE))[0]
out_acc_path  = STEP5_PATH + f'probe_accuracy_{base_filename}.json'

acc_report = {
    'source_file': EPR_RESULT_FILE,
    'overall_per_layer': overall_acc_by_layer,
    'per_emotion_per_layer': per_emotion_acc,
    'num_responses': len(flat_items),
}

with open(out_acc_path, 'w') as f:
    json.dump(acc_report, f, indent=2)

print(f"\nSaved: {out_acc_path}")

# Print a quick table of overall accuracy
print("\nLayer | Overall Accuracy")
print("------|-----------------")
for L in range(num_layers):
    print(f"  {L:2d}   | {overall_acc_by_layer[L]:.4f}")

# %%
# ── Save per-response predictions (optional, for error analysis) ─────────────
if per_response:
    pred_path = STEP5_PATH + f'per_response_predictions_{base_filename}.json'
    with open(pred_path, 'w') as f:
        json.dump(per_response, f, indent=2)
    print(f"Saved: {pred_path}")
