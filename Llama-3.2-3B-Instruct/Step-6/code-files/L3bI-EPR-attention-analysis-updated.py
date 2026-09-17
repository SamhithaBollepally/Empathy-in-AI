# %%
# ============================================================================
# Step 6 — Attention-Flow Analysis (per sample / emotion / layer / head)
#
# Generates a response to each sampled Set-2 utterance token-by-token and, at
# every step, records where attention went. Attention is split into four
# mutually exclusive buckets over the key positions, plus an "emotional" subset:
#
#   bos        - the <|begin_of_text|> token (the attention sink)
#   template   - system prompt + chat-template scaffolding
#   utterance  - the user's actual words
#   generated  - tokens the model has already produced this response
#   emotional  - SUBSET of `utterance`: tokens whose layer-11 state points in
#                the target emotion's direction (see emotion_scores_and_positions)
#
# bos + template + utterance + generated sums to ~1.0 per (layer, head), since
# each head's attention is a probability distribution. `emotional` is a subset of
# `utterance`, so it is NOT part of that sum.
#
# Two granularities are saved:
#   1. HEAD-LEVEL, for TRACK_LAYERS only, un-aggregated:
#        (total_steps, n_track_layers, n_heads, 5)
#      Lets you see individual head trajectories - e.g. "layer 11 head 7 attends
#      0.30 to emotional tokens at step 5, decaying to 0.05 by step 20".
#   2. LAYER-LEVEL, all 28 layers, averaged over heads:
#        (total_steps, 28, 5)
#      The overall flow profile through network depth.
#
# LAYER NUMBERING (important): out.attentions has 28 entries but
# out.hidden_states has 29 (embeddings + 28 decoder layers), so attentions[i]
# belongs to decoder layer i+1. TRACK_LAYERS is given in hidden-state numbering
# (1..28), matching PROBE_LAYER, and is converted to attention indices as L-1.
# The saved `layer_ids` array holds the hidden-state numbers so plots label
# correctly.
#
# Changes vs the previous version
# -------------------------------
# 1. INPUT IS SET 2 UTTERANCES (prompts_utterances_set2.json), reading
#    'first_utterance'. Responses are generated here, not read from a file, and
#    are saved alongside the analysis.
# 2. BOS/SINK IS ACCOUNTED FOR SEPARATELY rather than lumped into "prompt
#    attention", which previously inflated it and made the argmax token almost
#    always BOS.
# 3. EMOTIONAL TOKENS BY DIRECTION COSINE, not by applying the pooled
#    prompt-trained probe to single tokens (off-distribution: a demo showed BOS
#    classifies as 'angry' every time). The per-token probe output is still kept
#    per sample as a clearly labelled heuristic for eyeballing.
# 4. ENRICHMENT RATIO instead of a raw sum, which scaled with how many tokens
#    happened to be marked. NOTE: not stored per head - it is exactly derivable
#    from the emotional/utterance sums and the per-sample token counts, so
#    storing it would be redundant. See `enrichment()` in the loader notes.
# 5. Reproducible: HF_TOKEN env var, fixed torch seed.
# 6. Probe bundle: new tensor format (W/b/mu/sd), probe layer from the bundle's
#    CV-selected best_layer_l2.
# 7. MEMORY. Bucket reduction happens ON DEVICE, so only the reduced
#    (28, heads, 5) array crosses the PCIe bus per step instead of the full
#    (28, heads, seq) attention. Numeric arrays go to a compressed .npz as
#    float32 (4 bytes/value) rather than nested JSON lists of Python floats
#    (~32 bytes/value) - the boxed-float blow-up is what OOM'd at 200 samples.
#    Prefill attention is released immediately instead of staying pinned for the
#    whole generation loop via past_key_values.
# 8. MAX_PROMPT_TOKENS caps prompt length. Eager attention materialises a
#    (heads, L, L) matrix per layer during prefill, which grows QUADRATICALLY;
#    a single long utterance previously tried to allocate 4.44 GiB and OOM'd the
#    GPU. Over-long utterances are truncated and flagged, not skipped.
# 9. Per-sample OOM recovery: one bad sample is recorded and skipped rather than
#    killing the whole run.
# ============================================================================

# %%
import gc
import inspect
import json
import os
from collections import defaultdict

import huggingface_hub
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# %%
# ── Config ───────────────────────────────────────────────────────────────────
BASE_PATH = '/Volumes/Reading/Projects/Empathy-LLM/Llama-3.2-3B-Instruct/'
# BASE_PATH = '/content/drive/MyDrive/Project-4/'

DATA_PATH = BASE_PATH + 'Step-6/data/'

INPUT_FILE     = DATA_PATH + 'prompts_utterances_set2.json'
OUTPUT_FILE    = DATA_PATH + 'attention_flow_summary.json'   # metadata + aggregates
FLOW_FILE      = DATA_PATH + 'attention_flow_arrays.npz'     # numeric arrays
RESPONSES_FILE = DATA_PATH + 'attention_generated_responses.json'

N_SAMPLES_PER_EMOTION   = 20   # 20 x 10 emotions = 200 samples
ANALYSIS_MAX_NEW_TOKENS = 50   # token budget per sample

# Layers kept at HEAD level, in hidden-state numbering (1..28); converted to
# attention indices as L-1. Chosen to span the curve: early lexical (1, 5), the
# probe's best layer (11), the Fisher separability peak (13), late-middle
# (15, 20) and the final layers (27, 28). All 28 layers are still saved
# head-averaged, so nothing is lost at the layer level.
TRACK_LAYERS = (1, 5, 11, 13, 14, 15, 20, 27, 28)

# Eager attention is O(heads x L^2) per layer during prefill. 1024 keeps the
# worst case well inside an 80 GB A100; lower it if you see CUDA OOM.
MAX_PROMPT_TOKENS = 1024

EMO_COS_THRESHOLD = 0.10   # cosine to emotion direction required to mark a token
MAX_EMO_TOKENS    = 8      # cap on marked emotional tokens per prompt

# Per-sample token-level detail (prompt tokens, per-token cosines, heuristic
# probe labels) in the summary JSON. Useful for qualitative inspection.
KEEP_TOKEN_DETAIL = True

# Buckets are stored along the last axis of the numeric arrays, in this order.
BUCKET_ORDER = ('bos', 'template', 'utterance', 'emotional', 'generated')
EMO_I, UTT_I = BUCKET_ORDER.index('emotional'), BUCKET_ORDER.index('utterance')

SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# The CUDA OOM exception moved between torch namespaces across versions; fall
# back to RuntimeError (filtered on message) so recovery works either way.
OOM_ERROR = getattr(torch, 'OutOfMemoryError',
                    getattr(torch.cuda, 'OutOfMemoryError', RuntimeError))

# %%
# ── Authentication and model load ────────────────────────────────────────────
token = os.environ.get('HF_TOKEN')
if token:
    huggingface_hub.login(token=token)

device = torch.device('cuda' if torch.cuda.is_available()
                      else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {device}')

model_name = 'meta-llama/Llama-3.2-3B-Instruct'
tokenizer  = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = 'left'   # consistent with embedding extraction

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map='auto',
    attn_implementation='eager',  # required to return attention weights
)
model.eval()
MODEL_DEVICE = next(model.parameters()).device
print(f'Model loaded on {MODEL_DEVICE}')

# Restrict prefill logits to the final position when the installed transformers
# supports it. The kwarg was renamed, so probe rather than assume; if neither is
# present we simply pay the full (1, L, vocab) lm_head cost as before.
_FWD = inspect.signature(model.forward).parameters
_LOGITS_KW = next((k for k in ('logits_to_keep', 'num_logits_to_keep')
                   if k in _FWD), None)
PREFILL_KW = {_LOGITS_KW: 1} if _LOGITS_KW else {}
print(f'Prefill logits kwarg: {_LOGITS_KW or "unsupported (full logits)"}')

N_LAYERS = model.config.num_hidden_layers        # 28 decoder layers
N_HEADS  = model.config.num_attention_heads      # 24 heads
# hidden-state layer L  ->  attentions index L-1
TRACK_IDX = [L - 1 for L in TRACK_LAYERS]
assert all(0 <= i < N_LAYERS for i in TRACK_IDX), 'TRACK_LAYERS out of range'
print(f'{N_LAYERS} layers x {N_HEADS} heads; head-level layers: {list(TRACK_LAYERS)}')

# %%
# ── Load emotion probe + prototypes ──────────────────────────────────────────
bundle = torch.load(DATA_PATH + 'emotion_probes.pt', map_location='cpu',
                    weights_only=False)
EMOTION_CLASSES = [str(c) for c in bundle['classes']]
CLASS_TO_IDX    = {c: i for i, c in enumerate(EMOTION_CLASSES)}

PROBE_LAYER = bundle['best_layer_l2']          # CV-selected, currently 11
probe = bundle['layers'][PROBE_LAYER]['l2']
probe_W     = probe['W'].float()               # (3072, 10)
probe_b     = probe['b'].float()               # (10,)
probe_mu    = probe['mu'].float().squeeze(0)              # (3072,)
probe_sd    = probe['sd'].float().squeeze(0).clamp(min=1e-8)  # clamped to avoid /0

ctx_npz    = np.load(DATA_PATH + 'context_embeddings.npz', allow_pickle=True)
protos     = ctx_npz['protos']                 # (29, 10, 3072)
grand_mean = torch.tensor(ctx_npz['grand_mean'][PROBE_LAYER]).float()   # (3072,)
# Emotion direction per class = prototype - grand mean (contrastive direction).
emo_dirs = torch.tensor(protos[PROBE_LAYER]).float() - grand_mean   # (10, 3072)
emo_dirs = torch.nn.functional.normalize(emo_dirs, dim=1)

print(f'Probe layer {PROBE_LAYER}; {len(EMOTION_CLASSES)} emotions')

# %%
# ── Helpers ───────────────────────────────────────────────────────────────────
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


# Tokens the template itself costs, measured once so the utterance budget is exact.
TEMPLATE_OVERHEAD = len(tokenizer(format_chat(GEN_SYSTEM_PROMPT, ''),
                                  add_special_tokens=False)['input_ids'])
UTTERANCE_BUDGET  = MAX_PROMPT_TOKENS - TEMPLATE_OVERHEAD
assert UTTERANCE_BUDGET > 0, (
    f'MAX_PROMPT_TOKENS ({MAX_PROMPT_TOKENS}) must exceed template overhead '
    f'({TEMPLATE_OVERHEAD})'
)
print(f'Template overhead {TEMPLATE_OVERHEAD} tokens; '
      f'utterance budget {UTTERANCE_BUDGET} tokens')


def truncate_utterance(utterance):
    """Cap the utterance so the full prompt stays within MAX_PROMPT_TOKENS.

    Truncates from the end - emotional cues in these utterances are typically
    stated early. Returns (text, was_truncated, original_token_count).
    """
    ids = tokenizer(utterance, add_special_tokens=False)['input_ids']
    if len(ids) <= UTTERANCE_BUDGET:
        return utterance, False, len(ids)
    kept = tokenizer.decode(ids[:UTTERANCE_BUDGET], skip_special_tokens=True)
    return kept, True, len(ids)


def locate_token_buckets(chat, utterance, encoding, input_ids):
    """Classify each PROMPT position into a bucket.

    The chat string is built by inserting the utterance verbatim, so its char
    span is findable; offset mapping then maps chars -> token positions.

    Returns dict of position lists: bos / template / utterance.
    """
    raw_offsets = encoding['offset_mapping'][0]
    # offset_mapping is sometimes a list and sometimes a tensor depending on the
    # transformers / tokenizer version; handle both gracefully.
    offsets = (raw_offsets.cpu().tolist() if torch.is_tensor(raw_offsets)
               else list(raw_offsets))
    ids     = input_ids[0].tolist()

    bos = [0] if tokenizer.bos_token_id is not None and ids[0] == tokenizer.bos_token_id else []

    start = chat.find(utterance)
    utt = []
    if start >= 0:
        end = start + len(utterance)
        utt = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
    if not utt:   # fallback: boundary-mangled tokenisation
        utt = [i for i in range(len(ids)) if i not in bos]

    utterance_set = set(utt)
    template = [i for i in range(len(ids))
                if i not in utterance_set and i not in bos]
    return {'bos': bos, 'template': template, 'utterance': utt}


def emotion_scores_and_positions(h_layer, utterance_pos, emotion):
    """Direction-cosine detection on utterance tokens only.

    h_layer: (prompt_len, 3072) hidden states at PROBE_LAYER (prompt positions).
    Returns (emo_positions, scores-per-utterance-token).
    """
    target = CLASS_TO_IDX[emotion]
    h = h_layer.float().cpu()
    centred = h - grand_mean
    cos_all = torch.nn.functional.cosine_similarity(
        centred, emo_dirs[target].unsqueeze(0), dim=1)      # (prompt_len,)

    scored = sorted(((float(cos_all[p]), p) for p in utterance_pos),
                    key=lambda x: -x[0])
    emo = [p for s, p in scored if s >= EMO_COS_THRESHOLD][:MAX_EMO_TOKENS]
    return sorted(emo), {p: s for s, p in scored}


def probe_labels_for_tokens(h_layer):
    """HEURISTIC ONLY: per-token argmax of the pooled-trained probe.

    Kept as a secondary diagnostic - useful for eyeballing which words the probe
    'sees', but NOT used in any statistic (off-distribution for single tokens;
    BOS reproducibly classifies as 'angry').
    """
    h = h_layer.float().cpu()
    scores = ((h - probe_mu) / probe_sd) @ probe_W + probe_b
    return [EMOTION_CLASSES[i] for i in scores.argmax(dim=1).tolist()]


def sample_token(logits, temperature=0.7, top_p=0.9):
    """Sample one token using temperature + nucleus (top-p) filtering."""
    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum > top_p
    mask[:, 0] = False      # keep at least one token
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True) + 1e-8
    samples = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices.gather(1, samples)


def enrichment(emo_sum, utt_sum, n_emo, n_utt):
    """Attention per emotional token / attention per other utterance token.

    >1 means emotional tokens attract disproportionate attention. None when
    undefined (no emotional tokens, or no non-emotional tokens to compare to) -
    NOT when it is legitimately 0.0, which is itself a finding.
    """
    n_other = n_utt - n_emo
    if not n_emo or n_other <= 0:
        return None
    other_mean = (utt_sum - emo_sum) / n_other
    if other_mean <= 0:
        return None
    return (emo_sum / n_emo) / other_mean


# %%
# ── Per-step reduction (runs on device) ──────────────────────────────────────
def reduce_step(attn, idx, prompt_len):
    """Reduce one step's attention to bucket sums, on device.

    attn: (layers, heads, seq) for this generation step.
    idx:  {bucket: LongTensor of key positions} on the same device.

    Returns (bucket_sums, max_pos):
      bucket_sums (layers, heads, 5) float32 numpy, buckets in BUCKET_ORDER
      max_pos     int, the single most-attended position (layer/head averaged)

    Reducing here rather than after transfer means ~7 KB crosses the bus per
    step instead of the full (layers, heads, seq) attention. float32 first -
    summing fp16 over hundreds of positions loses precision.
    """
    a = attn.float()
    cols = []
    for name in BUCKET_ORDER:
        if name == 'generated':
            cols.append(a[:, :, prompt_len:].sum(2))   # empty slice -> 0 at step 0
        else:
            sel = idx[name]
            cols.append(a.index_select(2, sel).sum(2) if sel.numel()
                        else torch.zeros(a.shape[:2], device=a.device))
    bucket_sums = torch.stack(cols, dim=-1)            # (layers, heads, 5)
    max_pos = int(a.mean(dim=(0, 1)).argmax().item())
    return bucket_sums.cpu().numpy(), max_pos


class FlowAccumulator:
    """Running sums over generation steps, for the aggregate summary.

    O(layers x heads) state regardless of sample count. The per-step arrays are
    also written out in full (as float32), so these aggregates are a convenience
    - everything here is recomputable from the .npz.
    """

    def __init__(self):
        self.count = 0
        self.layer = None    # (28, 5)      head-averaged
        self.head  = None    # (n_track, heads, 5)

    def add(self, layer_bucket, head_bucket):
        if self.layer is None:
            self.layer = np.zeros_like(layer_bucket, dtype=np.float64)
            self.head  = np.zeros_like(head_bucket, dtype=np.float64)
        self.layer += layer_bucket
        self.head  += head_bucket
        self.count += 1

    @property
    def n(self):
        return max(self.count, 1)

    def layer_mean(self):
        return None if self.layer is None else self.layer / self.n

    def head_mean(self):
        return None if self.head is None else self.head / self.n


# %%
# ── Attention extraction for one sample ──────────────────────────────────────
def analyse_one(utterance, emotion, acc, emo_accs, max_new_tokens=50):
    """Manual token-by-token generation, capturing attention at every step.

    Returns (record, head_steps, layer_steps) where
      head_steps  list of (n_track, heads, 5) float32, one per generated token
      layer_steps list of (28, 5)             float32, one per generated token
    """
    emo_acc = emo_accs[emotion]
    utt_text, was_truncated, orig_tokens = truncate_utterance(utterance)
    chat    = format_chat(GEN_SYSTEM_PROMPT, utt_text)
    inputs  = tokenizer(chat, return_tensors='pt',
                        return_offsets_mapping=True).to(MODEL_DEVICE)
    prompt_len     = inputs['input_ids'].shape[1]
    input_ids      = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    buckets       = locate_token_buckets(chat, utt_text, inputs, input_ids)
    generated_ids = []

    # ── Prefill ──────────────────────────────────────────────────────────────
    # PREFILL_KW asks for logits at the last position only; otherwise the
    # lm_head runs over all L positions to produce a (1, L, 128256) tensor we
    # keep one row of.
    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
            **PREFILL_KW,
        )
    # Attention for the first generated token is the last query row of each
    # layer. Stacked on device; `del out` below frees the full
    # (layers, heads, L, L) prefill attention immediately - it previously stayed
    # pinned on the GPU for the whole generation loop via past_key_values.
    step_attn = torch.stack([a[0, :, -1, :] for a in out.attentions])

    # Hidden states at PROBE_LAYER for prompt tokens: both detectors use these.
    h_probe = out.hidden_states[PROBE_LAYER][0, :prompt_len, :].detach()
    emo_positions, emo_scores = emotion_scores_and_positions(
        h_probe, buckets['utterance'], emotion)
    token_probe_labels = probe_labels_for_tokens(h_probe)

    # Key-position index tensors, built once per sample and reused every step.
    idx = {k: torch.as_tensor(v, dtype=torch.long, device=step_attn.device)
           for k, v in buckets.items()}
    idx['emotional'] = torch.as_tensor(emo_positions, dtype=torch.long,
                                       device=step_attn.device)
    n_emo, n_utt = len(emo_positions), len(buckets['utterance'])
    emo_set = set(emo_positions)      # for efficient max-attention bucket lookup

    past_key_values = out.past_key_values
    next_token      = sample_token(out.logits[:, -1, :])
    generated_ids.append(next_token.item())
    del out, h_probe

    head_steps, layer_steps, step_stats = [], [], []

    def record_step(attn, t):
        bh, max_pos = reduce_step(attn, idx, prompt_len)   # (layers, heads, 5)
        layer_bucket = bh.mean(axis=1)                     # (28, 5) head-averaged
        head_bucket  = bh[TRACK_IDX]                       # (n_track, heads, 5)
        acc.add(layer_bucket, head_bucket)
        emo_acc.add(layer_bucket, head_bucket)
        layer_steps.append(layer_bucket.astype(np.float32))
        head_steps.append(head_bucket.astype(np.float32))

        # Scalars: mean over layers and heads, i.e. the whole-model view.
        overall = bh.mean(axis=(0, 1))                     # (5,)
        e = enrichment(float(overall[EMO_I]), float(overall[UTT_I]), n_emo, n_utt)
        step_stats.append({
            'step': t,
            **{f'{b}_attention': round(float(overall[i]), 4)
               for i, b in enumerate(BUCKET_ORDER)},
            'emotional_enrichment': None if e is None else round(e, 4),
            'max_attention_pos': max_pos,   # -> token string after decode
            'max_attention_bucket': ('emotional' if max_pos in emo_set else
                                     'bos' if max_pos in set(buckets['bos']) else
                                     'template' if max_pos in set(buckets['template'])
                                     else 'utterance' if max_pos < prompt_len
                                     else 'generated'),
        })

    record_step(step_attn, 0)
    del step_attn

    # ── Autoregressive generation ─────────────────────────────────────────────
    for t in range(1, max_new_tokens):
        if next_token.item() == tokenizer.eos_token_id:
            break
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=torch.long, device=MODEL_DEVICE)],
            dim=1
        )
        with torch.inference_mode():
            out = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
            )
        # out.attentions: tuple of (1, heads, 1, prompt_len + t) per layer
        step_attn       = torch.stack([a[0, :, 0, :] for a in out.attentions])
        past_key_values = out.past_key_values
        next_token      = sample_token(out.logits[:, -1, :])
        generated_ids.append(next_token.item())
        del out
        record_step(step_attn, t)
        del step_attn

    # ── Decode ────────────────────────────────────────────────────────────────
    full_ids         = input_ids[0].cpu().tolist() + generated_ids
    all_tokens       = [tokenizer.decode([tid], skip_special_tokens=False) for tid in full_ids]
    generated_tokens = all_tokens[prompt_len:]
    # Step t's attention is the attention that produced generated token t.
    for t, s in enumerate(step_stats):
        s['generated_token']     = generated_tokens[t]
        s['max_attention_token'] = all_tokens[s.pop('max_attention_pos')]

    # empty_cache() is deliberately NOT called per sample - flushing the cache
    # every sample makes the allocator re-request blocks. Done periodically in
    # the run loop instead.
    del inputs, past_key_values

    record = {
        'emotion':             emotion,
        'utterance':           utterance,
        # Decoded from ids so EOS/special markers are dropped - joining the
        # skip_special_tokens=False strings left '<|eot_id|>' in the text.
        'generated_response':  tokenizer.decode(generated_ids,
                                                skip_special_tokens=True).strip(),
        'prompt_len':          prompt_len,
        'gen_len':             len(generated_ids),
        'n_utterance_tokens':  n_utt,
        'n_template_tokens':   len(buckets['template']),
        'n_emotional_tokens':  n_emo,
        'emotional_positions': emo_positions,
        'was_truncated':       was_truncated,
        'utterance_tokens_original': orig_tokens,
        'step_stats':          step_stats,
    }
    if KEEP_TOKEN_DETAIL:
        record.update({
            'prompt_tokens':    all_tokens[:prompt_len],
            'generated_tokens': generated_tokens,
            # utterance-token cosine to the emotion direction (detection evidence)
            'emo_token_cosines': {str(p): round(s, 4) for p, s in emo_scores.items()},
            # HEURISTIC diagnostic only - see probe_labels_for_tokens docstring
            'token_probe_labels': {str(p): token_probe_labels[p]
                                   for p in buckets['utterance']},
        })
    return record, head_steps, layer_steps


# %%
# ── Load Set-2 utterances and build sample list ───────────────────────────────
with open(INPUT_FILE) as f:
    data = json.load(f)

samples = []
for emotion, entries in data.items():
    for it in entries[:N_SAMPLES_PER_EMOTION]:
        samples.append((emotion, it['first_utterance']))

print(f"Analysing {len(samples)} samples "
      f"({N_SAMPLES_PER_EMOTION}/emotion) from:\n  {INPUT_FILE}")

# %%
# ── Run analysis sample by sample ────────────────────────────────────────────
ACC      = FlowAccumulator()
EMO_ACCS = defaultdict(FlowAccumulator)

per_sample   = []
head_chunks  = []   # per step: (n_track, heads, 5) float32
layer_chunks = []   # per step: (28, 5)             float32
step_sample  = []   # per step: index into per_sample
step_num     = []   # per step: step number within its sample
failures     = []

for idx_s, (emotion, utterance) in enumerate(samples):
    print(f"  [{idx_s + 1}/{len(samples)}] {emotion} — {utterance[:60]}...")
    try:
        rec, hsteps, lsteps = analyse_one(
            utterance, emotion, ACC, EMO_ACCS,
            max_new_tokens=ANALYSIS_MAX_NEW_TOKENS)
    except OOM_ERROR as err:
        if 'out of memory' not in str(err).lower():
            raise            # a real bug, not a capacity problem
        # One oversized sample should not lose the whole run.
        failures.append({'emotion': emotion, 'utterance': utterance,
                         'error': 'CUDA OOM', 'detail': str(err)[:200]})
        print('      SKIPPED (out of memory)')
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        continue

    s_i = len(per_sample)
    per_sample.append(rec)
    head_chunks.extend(hsteps)
    layer_chunks.extend(lsteps)
    step_sample.extend([s_i] * len(hsteps))
    step_num.extend(range(len(hsteps)))

    if idx_s % 25 == 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if not per_sample:
    raise RuntimeError('No samples analysed - check INPUT_FILE and config.')
print(f'\nDone: {len(per_sample)} samples, {len(head_chunks)} generation steps, '
      f'{len(failures)} failures')

# %%
# ── Save numeric arrays (.npz, float32) ──────────────────────────────────────
# float32 in a binary array is 4 bytes/value; the same numbers as Python floats
# nested in JSON cost ~32 bytes each, which is what exhausted RAM previously.
head_bucket  = np.stack(head_chunks)    # (T, n_track, heads, 5)
layer_bucket = np.stack(layer_chunks)   # (T, 28, 5)

np.savez_compressed(
    FLOW_FILE,
    head_bucket=head_bucket,
    layer_bucket=layer_bucket,
    step_sample=np.asarray(step_sample, dtype=np.int32),
    step_num=np.asarray(step_num, dtype=np.int32),
    bucket_order=np.asarray(BUCKET_ORDER),
    track_layers=np.asarray(TRACK_LAYERS, dtype=np.int32),
    layer_ids=np.arange(1, N_LAYERS + 1, dtype=np.int32),   # hidden-state numbering
    sample_emotion=np.asarray([r['emotion'] for r in per_sample]),
    sample_gen_len=np.asarray([r['gen_len'] for r in per_sample], dtype=np.int32),
    sample_prompt_len=np.asarray([r['prompt_len'] for r in per_sample], dtype=np.int32),
    sample_n_emo=np.asarray([r['n_emotional_tokens'] for r in per_sample], dtype=np.int32),
    sample_n_utt=np.asarray([r['n_utterance_tokens'] for r in per_sample], dtype=np.int32),
)
print(f'Saved: {FLOW_FILE}')
print(f'  head_bucket  {head_bucket.shape}  {head_bucket.nbytes / 1e6:.1f} MB raw')
print(f'  layer_bucket {layer_bucket.shape}  {layer_bucket.nbytes / 1e6:.1f} MB raw')

# %%
# ── Aggregate summary ─────────────────────────────────────────────────────────
def _mean(arr):
    return round(float(np.mean(arr)), 4) if len(arr) else None


def _layer_profile(acc):
    """Head-averaged mean attention per bucket, per layer -> {bucket: [28]}."""
    m = acc.layer_mean()
    if m is None:
        return {b: [] for b in BUCKET_ORDER}
    return {b: [round(float(x), 5) for x in m[:, i]]
            for i, b in enumerate(BUCKET_ORDER)}


def _head_profile(acc):
    """Per-tracked-layer, per-head mean -> {bucket: [[heads] per track layer]}."""
    m = acc.head_mean()
    if m is None:
        return {b: [] for b in BUCKET_ORDER}
    return {b: [[round(float(v), 5) for v in m[li, :, i]]
                for li in range(len(TRACK_LAYERS))]
            for i, b in enumerate(BUCKET_ORDER)}


# Per-step scalars pooled for the headline numbers.
overall     = defaultdict(list)
per_emotion = defaultdict(lambda: defaultdict(list))
for r in per_sample:
    e = r['emotion']
    for step in r['step_stats']:
        for b in BUCKET_ORDER:
            k = f'{b}_attention'
            overall[k].append(step[k])
            per_emotion[e][k].append(step[k])
        if step['emotional_enrichment'] is not None:
            overall['enrichment'].append(step['emotional_enrichment'])
            per_emotion[e]['enrichment'].append(step['emotional_enrichment'])

# Rankings based on mean attention to emotional tokens. At only 20 samples per
# emotion these per-head means are noisy, so treat these as shortlists to
# investigate, not statistical findings. Consistency across samples must be
# checked before claiming any layer or head 'specialises'.
head_emo = ACC.head_mean()[:, :, EMO_I]      # (n_track, heads)
layer_emo = ACC.layer_mean()[:, EMO_I]       # (28,)

# Top (layer, head) pairs globally across tracked layers.
flat     = np.argsort(head_emo.ravel())[::-1]
top_heads = [{'layer': int(TRACK_LAYERS[i // N_HEADS]), 'head': int(i % N_HEADS),
              'mean_emotional_attention': round(float(head_emo.ravel()[i]), 5)}
             for i in flat[:10]]

# Heads ranked within each tracked layer.
head_rank_by_layer = {
    int(TRACK_LAYERS[li]): [
        {'rank': int(rank + 1), 'head': int(hi),
         'mean_emotional_attention': round(float(val), 5)}
        for rank, (hi, val) in enumerate(
            sorted(enumerate(head_emo[li]), key=lambda x: -x[1]))
    ]
    for li in range(len(TRACK_LAYERS))
}

# All 28 layers ranked by their head-averaged emotional attention.
layer_rank = [
    {'rank': int(rank + 1), 'layer': int(layer_id),
     'mean_emotional_attention': round(float(val), 5)}
    for rank, (layer_id, val) in enumerate(
        sorted(enumerate(layer_emo, start=1), key=lambda x: -x[1]))
]

summary = {
    'config': {
        'input_file':            INPUT_FILE,
        'flow_arrays':           FLOW_FILE,
        'seed':                  SEED,
        'n_samples_per_emotion': N_SAMPLES_PER_EMOTION,
        'max_new_tokens':        ANALYSIS_MAX_NEW_TOKENS,
        'max_prompt_tokens':     MAX_PROMPT_TOKENS,
        'probe_layer':           int(PROBE_LAYER),
        'emo_cos_threshold':     EMO_COS_THRESHOLD,
        'max_emo_tokens':        MAX_EMO_TOKENS,
        'track_layers':          list(TRACK_LAYERS),
        'bucket_order':          list(BUCKET_ORDER),
        'n_layers':              N_LAYERS,
        'n_heads':               N_HEADS,
        'layer_numbering':       'hidden-state (1..28); attentions[i] is layer i+1',
    },
    'counts': {
        'n_samples':      len(per_sample),
        'n_total_steps':  ACC.count,
        'n_truncated':    sum(r['was_truncated'] for r in per_sample),
        'n_failures':     len(failures),
        'mean_emotional_tokens': _mean([r['n_emotional_tokens'] for r in per_sample]),
        'mean_utterance_tokens': _mean([r['n_utterance_tokens'] for r in per_sample]),
    },
    'overall': {
        **{f'mean_{b}_attention': _mean(overall[f'{b}_attention'])
           for b in BUCKET_ORDER},
        # enrichment > 1 = emotional tokens attract disproportionate attention.
        # n_enrichment_steps < n_total_steps means some steps had no emotional
        # token (or no comparison token), so it was undefined there - needed to
        # read the mean honestly.
        'mean_emotional_enrichment': _mean(overall['enrichment']),
        'n_enrichment_steps':        len(overall['enrichment']),
    },
    # Head-averaged flow through all 28 layers.
    'layer_profile': _layer_profile(ACC),
    # Per-head detail at the tracked layers, averaged over samples and steps.
    'head_profile_tracked_layers': _head_profile(ACC),
    'top_emotional_heads': top_heads,
    'head_rank_by_layer': head_rank_by_layer,
    'layer_rank': layer_rank,
    'per_emotion': {
        e: {
            **{f'mean_{b}_attention': _mean(v[f'{b}_attention'])
               for b in BUCKET_ORDER},
            'mean_emotional_enrichment': _mean(v['enrichment']),
            'n_steps':                   EMO_ACCS[e].count,
            'layer_profile':             _layer_profile(EMO_ACCS[e]),
            'head_profile_tracked_layers': _head_profile(EMO_ACCS[e]),
        }
        for e, v in per_emotion.items()
    },
    'failures': failures,
}

# %%
# ── Save ──────────────────────────────────────────────────────────────────────
with open(OUTPUT_FILE, 'w') as f:
    json.dump({'summary': summary, 'per_sample': per_sample}, f, indent=2)
print(f'Saved: {OUTPUT_FILE}')

# Generated responses saved separately - usable as inputs for later steps.
responses = defaultdict(list)
for r in per_sample:
    responses[r['emotion']].append({
        'utterance': r['utterance'],
        'response':  r['generated_response'],
    })
with open(RESPONSES_FILE, 'w') as f:
    json.dump(dict(responses), f, indent=2)
print(f'Saved: {RESPONSES_FILE}')

print('\n── Overall ──')
print(json.dumps(summary['overall'], indent=2))
print('\n── Top emotion-attending heads (layer, head) ──')
for h in top_heads[:5]:
    print(f"  layer {h['layer']:>2} head {h['head']:>2}: "
          f"{h['mean_emotional_attention']:.4f}")
print('\n── Top emotion-attending layers (head-averaged) ──')
for r in layer_rank[:5]:
    print(f"  layer {r['layer']:>2}: {r['mean_emotional_attention']:.4f}")
