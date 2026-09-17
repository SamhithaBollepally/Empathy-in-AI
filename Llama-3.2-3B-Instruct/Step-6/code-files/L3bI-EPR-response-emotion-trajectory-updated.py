# %%
# ============================================================================
# Step 6b — Generated-Token Emotion Trajectory Analysis
#
# Tests the project's central claim: the model ENCODES emotion while reading the
# input, but does it still carry that emotion when GENERATING the response?
#
# Three measurements per sample
# -----------------------------
# 1. input_pooled     - the raw utterance, masked-mean pooled, cosine to each
#                       emotion direction.
# 2. response_pooled  - the raw generated response, pooled the SAME way.
#                       input_pooled vs response_pooled is the like-for-like
#                       comparison, and their difference is the headline
#                       `drift` metric.
# 3. token_trajectory - each response token's hidden state IN THE ACTUAL
#                       GENERATION CONTEXT (templated prompt + response so far),
#                       cosine to each emotion direction. Shows the dynamics
#                       over time.
#
# COMPARABILITY (important). context_embeddings.npz was built by
# L3bI-Datapreprocessing-updated.py with:
#     prompt_format = 'raw_text_no_chat_template'
#     pooling       = 'masked_mean_bos_excluded'
# so anything compared against those prototypes must be pooled the SAME way:
# raw text, no chat template, mean over real tokens with BOS excluded. That is
# why (1) and (2) re-embed the raw strings rather than reusing the templated
# prompt - mean-pooling the templated prompt would mostly encode the system
# prompt, since template+BOS is ~80% of those tokens.
#
# For the same reason, pooled cosines (1, 2) and per-token cosines (3) are NOT
# on the same scale: pooling averages out per-token noise and systematically
# raises cosine magnitude. Compare pooled-with-pooled, and read the trajectory
# for its SHAPE over time, not against the pooled numbers.
#
# Cheap and safe: no attention weights, so no O(L^2) matrices. The pooled
# embeddings are batched; only the trajectory needs one forward pass per sample.
#
# Output: response_emotion_trajectories.json
# ============================================================================

# %%
import gc
import json
import os

import huggingface_hub
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# %%
# ── Config ───────────────────────────────────────────────────────────────────
BASE_PATH = '/Volumes/Reading/Projects/Empathy-LLM/Llama-3.2-3B-Instruct/'
# BASE_PATH = '/content/drive/MyDrive/Project-4/'

DATA_PATH = BASE_PATH + 'Step-6/data/'

# Source: per-sample records saved by the attention-flow script (utterance +
# generated_response), so both analyses describe exactly the same responses.
ATTENTION_SUMMARY_FILE = DATA_PATH + 'attention_flow_summary.json'
OUTPUT_FILE            = DATA_PATH + 'response_emotion_trajectories.json'

# Must match L3bI-Datapreprocessing-updated.py so pooled vectors land in the
# same space as the prototypes.
MAX_LENGTH = 512
BATCH_SIZE = 32

SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# %%
# ── Model & tokenizer (no eager attention needed - we never read attention) ──
token = os.environ.get('HF_TOKEN')
if token:
    huggingface_hub.login(token=token)

MODEL_NAME = 'meta-llama/Llama-3.2-3B-Instruct'
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
# 'right' matches the preprocessing script. It also keeps BOS at index 0, which
# build_pool_mask relies on to drop it.
tokenizer.padding_side = 'right'

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map='auto',
)
model.eval()
MODEL_DEVICE = next(model.parameters()).device
print(f'Model loaded on {MODEL_DEVICE}')

# %%
# ── Probe bundle + emotion directions ────────────────────────────────────────
bundle = torch.load(DATA_PATH + 'emotion_probes.pt', map_location='cpu',
                    weights_only=False)
EMOTION_CLASSES = [str(c) for c in bundle['classes']]
CLASS_TO_IDX    = {c: i for i, c in enumerate(EMOTION_CLASSES)}
PROBE_LAYER     = bundle['best_layer_l2']

ctx_npz    = np.load(DATA_PATH + 'context_embeddings.npz', allow_pickle=True)
protos     = ctx_npz['protos']                                   # (29, 10, 3072)
PROTO_EMOTIONS = [str(e) for e in ctx_npz['emotions']]
# The probe bundle and the prototype file must agree on class order, or every
# cosine would be attributed to the wrong emotion.
assert PROTO_EMOTIONS == EMOTION_CLASSES, (
    f'class order mismatch:\n  probes: {EMOTION_CLASSES}\n  protos: {PROTO_EMOTIONS}')

grand_mean = torch.tensor(ctx_npz['grand_mean'][PROBE_LAYER]).float().to(MODEL_DEVICE)
emo_dirs   = torch.tensor(protos[PROBE_LAYER]).float().to(MODEL_DEVICE) - grand_mean
emo_dirs   = torch.nn.functional.normalize(emo_dirs, dim=1)      # (10, 3072)

print(f'Probe layer {PROBE_LAYER}; emotions: {EMOTION_CLASSES}')

# %%
# ── Pooling, copied from L3bI-Datapreprocessing-updated.py ───────────────────
def build_pool_mask(input_ids, attention_mask):
    """attention_mask with the BOS token dropped."""
    mask = attention_mask.clone()
    if tokenizer.bos_token_id is not None:
        mask[:, 0] = torch.where(input_ids[:, 0] == tokenizer.bos_token_id,
                                 torch.zeros_like(mask[:, 0]), mask[:, 0])
    empty = mask.sum(dim=1) == 0     # a BOS-only text would be all-zero
    if empty.any():
        mask[empty] = attention_mask[empty]
    return mask


def masked_mean_pool(hidden_state, pool_mask):
    """Mean over real (non-pad, non-BOS) tokens, accumulated in float32."""
    h = hidden_state.float()
    m = pool_mask.unsqueeze(-1).float()
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def cosine_to_emotions(h):
    """h: (..., 3072) -> (..., n_emotions) cosine to each emotion direction.

    emo_dirs is already unit-norm; cosine_similarity normalises h itself, so no
    pre-normalisation is needed here.
    """
    centred = h - grand_mean
    return torch.nn.functional.cosine_similarity(
        centred.unsqueeze(-2), emo_dirs.unsqueeze(0), dim=-1)


def embed_pooled(texts, desc=''):
    """Masked-mean pooled hidden state at PROBE_LAYER for raw texts.

    Same recipe as the preprocessing script, so results are directly comparable
    to the prototypes. Returns (len(texts), 3072) float32 on MODEL_DEVICE.
    """
    out = torch.empty((len(texts), model.config.hidden_size),
                      dtype=torch.float32, device=MODEL_DEVICE)
    n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    for b, start in enumerate(range(0, len(texts), BATCH_SIZE)):
        batch = texts[start:start + BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                           truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(MODEL_DEVICE) for k, v in inputs.items()}
        pool_mask = build_pool_mask(inputs['input_ids'], inputs['attention_mask'])
        with torch.inference_mode():
            # model.model -> base transformer, skips the unused lm_head.
            hs = model.model(**inputs, output_hidden_states=True,
                             return_dict=True).hidden_states[PROBE_LAYER]
        out[start:start + len(batch)] = masked_mean_pool(hs, pool_mask)
        del hs, inputs, pool_mask
        if (b + 1) % 10 == 0 or b + 1 == n_batches:
            print(f'  {desc} batch {b + 1}/{n_batches}')
    return out


# %%
# ── Generation context, identical to the attention-flow script ───────────────
# This string must match L3bI-EPR-attention-analysis-updated.py exactly, or the
# token trajectory would be measured in a context the response was not produced
# in.
GEN_SYSTEM_PROMPT = (
    'You are a medical assistant. Given the user\'s emotional situation, do two things:\n'
    '1. Write a response (1-2 sentences).\n'
    '2. Rate how emotional is the user\'s situation on a scale of 1 (mild) to 5 (highly emotional).\n\n'
    'Format your output exactly as:\n'
    'Response: <your response>\n'
    'Intensity: <1-5>'
)


def format_chat(system, user):
    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})
    messages.append({'role': 'user', 'content': user})
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)


def token_trajectory(utterance, response, emotion):
    """Per-response-token emotion cosines, in the real generation context."""
    prompt_ids = tokenizer(format_chat(GEN_SYSTEM_PROMPT, utterance),
                           add_special_tokens=False)['input_ids']
    resp_ids   = tokenizer(response, add_special_tokens=False)['input_ids']
    if not resp_ids:
        return []

    ids = torch.tensor([prompt_ids + resp_ids], dtype=torch.long,
                       device=MODEL_DEVICE)
    with torch.inference_mode():
        hs = model.model(ids, output_hidden_states=True,
                         return_dict=True).hidden_states[PROBE_LAYER]
    resp_h = hs[0, len(prompt_ids):].float()            # (resp_len, 3072)
    scores = cosine_to_emotions(resp_h).cpu().numpy()   # (resp_len, n_emotions)
    del hs, resp_h, ids

    ti = CLASS_TO_IDX[emotion]
    return [
        {'step': t,
         'token': tokenizer.decode([resp_ids[t]], skip_special_tokens=True),
         'target_emotion': round(float(scores[t, ti]), 5),
         'all_emotions': {c: round(float(scores[t, i]), 5)
                          for i, c in enumerate(EMOTION_CLASSES)}}
        for t in range(len(resp_ids))
    ]


# %%
# ── Load the generated responses ─────────────────────────────────────────────
with open(ATTENTION_SUMMARY_FILE) as f:
    records = json.load(f)['per_sample']

# Drop samples with an empty response - they carry no output signal.
records = [r for r in records if r['generated_response'].strip()]
emotions   = [r['emotion'] for r in records]
utterances = [r['utterance'] for r in records]
responses  = [r['generated_response'] for r in records]
print(f'{len(records)} samples with non-empty responses from {ATTENTION_SUMMARY_FILE}')

# %%
# ── 1 & 2: pooled input and response embeddings (batched) ────────────────────
print('Embedding raw utterances...')
input_vecs = embed_pooled(utterances, desc='utterances')
print('Embedding raw responses...')
resp_vecs  = embed_pooled(responses, desc='responses')

input_cos = cosine_to_emotions(input_vecs).cpu().numpy()   # (N, 10)
resp_cos  = cosine_to_emotions(resp_vecs).cpu().numpy()    # (N, 10)
del input_vecs, resp_vecs
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# %%
# ── 3: per-token trajectories ────────────────────────────────────────────────
per_sample = []
for i, rec in enumerate(records):
    emotion = emotions[i]
    ti = CLASS_TO_IDX[emotion]
    print(f'  [{i + 1}/{len(records)}] {emotion} — {responses[i][:60]}...')

    traj = token_trajectory(utterances[i], responses[i], emotion)
    target_curve = np.array([s['target_emotion'] for s in traj], dtype=np.float64)

    # Rank of the target emotion among all 10 for the pooled response: rank 1
    # means the response most resembles the emotion the input expressed.
    resp_rank = int((resp_cos[i] > resp_cos[i, ti]).sum()) + 1
    input_rank = int((input_cos[i] > input_cos[i, ti]).sum()) + 1

    per_sample.append({
        'emotion':   emotion,
        'utterance': utterances[i],
        'response':  responses[i],
        # like-for-like: both pooled the same way, both comparable to prototypes
        'input_pooled':    {c: round(float(input_cos[i, j]), 5)
                            for j, c in enumerate(EMOTION_CLASSES)},
        'response_pooled': {c: round(float(resp_cos[i, j]), 5)
                            for j, c in enumerate(EMOTION_CLASSES)},
        'input_target_cos':    round(float(input_cos[i, ti]), 5),
        'response_target_cos': round(float(resp_cos[i, ti]), 5),
        'drift':               round(float(input_cos[i, ti] - resp_cos[i, ti]), 5),
        'input_argmax':    EMOTION_CLASSES[int(input_cos[i].argmax())],
        'response_argmax': EMOTION_CLASSES[int(resp_cos[i].argmax())],
        'input_target_rank':    input_rank,
        'response_target_rank': resp_rank,
        # dynamics: NOT on the same scale as the pooled numbers above
        'resp_len':            len(traj),
        'traj_mean_target':    round(float(target_curve.mean()), 5) if len(traj) else None,
        'traj_std_target':     round(float(target_curve.std(ddof=0)), 5) if len(traj) else None,
        'traj_first_target':   round(float(target_curve[0]), 5) if len(traj) else None,
        'traj_last_target':    round(float(target_curve[-1]), 5) if len(traj) else None,
        'token_trajectory':    traj,
    })

    if i % 50 == 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# %%
# ── Aggregate per emotion ────────────────────────────────────────────────────
def pad_nan(arrays):
    """Stack ragged 1-D arrays into a NaN-padded matrix for nan-aware stats."""
    max_len = max(len(a) for a in arrays)
    out = np.full((len(arrays), max_len), np.nan)
    for i, a in enumerate(arrays):
        out[i, :len(a)] = a
    return out


per_emotion = {}
for emotion in EMOTION_CLASSES:
    rows = [r for r in per_sample if r['emotion'] == emotion]
    if not rows:
        continue
    curves = [np.array([s['target_emotion'] for s in r['token_trajectory']])
              for r in rows if r['token_trajectory']]
    in_cos   = np.array([r['input_target_cos'] for r in rows])
    out_cos  = np.array([r['response_target_cos'] for r in rows])

    entry = {
        'n_samples':              len(rows),
        'mean_input_target_cos':    round(float(in_cos.mean()), 5),
        'std_input_target_cos':     round(float(in_cos.std(ddof=0)), 5),
        'mean_response_target_cos': round(float(out_cos.mean()), 5),
        'std_response_target_cos':  round(float(out_cos.std(ddof=0)), 5),
        # >0 = response carries LESS of the target emotion than the input did
        'mean_drift':             round(float((in_cos - out_cos).mean()), 5),
        # how often the pooled vector's top emotion is the labelled one
        'input_argmax_accuracy':    round(
            float(np.mean([r['input_argmax'] == emotion for r in rows])), 4),
        'response_argmax_accuracy': round(
            float(np.mean([r['response_argmax'] == emotion for r in rows])), 4),
        'mean_response_target_rank': round(
            float(np.mean([r['response_target_rank'] for r in rows])), 3),
    }
    if curves:
        mat = pad_nan(curves)
        entry.update({
            'mean_trajectory':   [round(float(v), 5) for v in np.nanmean(mat, axis=0)],
            'std_trajectory':    [round(float(v), 5) for v in np.nanstd(mat, axis=0)],
            'n_at_step':         [int(v) for v in (~np.isnan(mat)).sum(axis=0)],
        })
    per_emotion[emotion] = entry

# %%
# ── Save ─────────────────────────────────────────────────────────────────────
output = {
    'config': {
        'source_file':     ATTENTION_SUMMARY_FILE,
        'model':           MODEL_NAME,
        'probe_layer':     int(PROBE_LAYER),
        'n_samples':       len(per_sample),
        'emotion_classes': EMOTION_CLASSES,
        'pooling':         'masked_mean_bos_excluded (matches prototypes)',
        'pooled_text_format': 'raw_text_no_chat_template (matches prototypes)',
        'trajectory_context': 'templated prompt + response (real generation context)',
        'note': ('pooled cosines and per-token cosines are not on the same '
                 'scale; compare pooled with pooled'),
    },
    'per_sample':  per_sample,
    'per_emotion': per_emotion,
}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2)
print(f'Saved: {OUTPUT_FILE}')

# %%
# ── Report ───────────────────────────────────────────────────────────────────
all_drift = np.array([r['drift'] for r in per_sample])
print('\n── Overall ──')
print(f'  mean input  target cos: {np.mean([r["input_target_cos"] for r in per_sample]):.4f}')
print(f'  mean response target cos: {np.mean([r["response_target_cos"] for r in per_sample]):.4f}')
print(f'  mean drift (input - response): {all_drift.mean():+.4f}')
print(f'  input  argmax accuracy: '
      f'{np.mean([r["input_argmax"] == r["emotion"] for r in per_sample]):.3f}')
print(f'  response argmax accuracy: '
      f'{np.mean([r["response_argmax"] == r["emotion"] for r in per_sample]):.3f}')
print('  (argmax accuracy vs 0.100 chance for 10 emotions)')

print('\n── Per emotion (drift > 0 = response carries less of the emotion) ──')
for emotion, agg in sorted(per_emotion.items(), key=lambda x: -x[1]['mean_drift']):
    print(f"  {emotion:12s} input {agg['mean_input_target_cos']:+.4f}  "
          f"response {agg['mean_response_target_cos']:+.4f}  "
          f"drift {agg['mean_drift']:+.4f}  "
          f"resp_argmax_acc {agg['response_argmax_accuracy']:.2f}")
