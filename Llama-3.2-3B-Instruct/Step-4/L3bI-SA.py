# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %%
# ============================================================================
# Steering Activation (SA): congruent emotion steering during generation.
#
# For an utterance of emotion e (e.g. "sad"), we ADD that emotion's
# context-embedding prototype into the residual stream at a SINGLE layer, then
# generate the response — forcing it toward emotion e (congruent steering).
#
#   hidden += alpha * (prototype / ||prototype||)     # raw unit-normalized
#
# Experiment: one layer at a time (clean layer attribution) over a chosen set
# of layers, sweeping alpha to find the stable range before scaling up.
# ============================================================================
import os
import json
import gc
import getpass
import numpy as np
import torch
import huggingface_hub
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

# %%
BASE_PATH = '/Volumes/Reading/Projects/Empathy-LLM/Llama-3.2-3B-Instruct/'
#BASE_PATH = '/content/drive/MyDrive/Project-4/'

EMOTIONS = ['afraid', 'angry', 'anxious', 'devastated', 'lonely', 'sad',
            'terrified', 'furious', 'grateful', 'hopeful', 'faithful']

STEP1_PATH = BASE_PATH + 'Step-1/'
STEP4_PATH = BASE_PATH + 'Step-4/'
os.makedirs(STEP4_PATH, exist_ok=True)

# ---- Steering experiment config ----------------------------------------------
# Layers use the hidden_states / context_embeddings convention:
#   0 = token-embedding output, 1..28 = output of decoder block (i-1).
# "Steer layer L" == add to the residual stream AFTER model.model.layers[L-1].
STEER_LAYERS = [1, 5, 8, 10, 15, 20, 25, 27, 28]

# Norm-matched steering: the push is a FRACTION of each hidden state's own norm,
#   hidden += frac * ||hidden|| * unit_vec
# so the RELATIVE nudge is comparable across layers (Llama's hidden-state norms
# grow with depth). 0.2 == a 20% nudge. Set STEER_MODE='fixed' to instead use an
# absolute magnitude (hidden += coeff * unit_vec).
STEER_MODE = 'norm_matched'
COEFFS     = [0.05, 0.1, 0.2, 0.3, 0.5]   # fractions of ||hidden||; start ~0.2
VECTOR_TYPE  = 'raw_unit'            # raw prototype, unit-normalized (contrastive later)
TOKEN_POSITIONS = 'all'              # steer all positions during generation

# For the initial sweep keep it cheap: steer only the first N utterances per
# emotion. Set to None to run the full set once a stable alpha is chosen.
SWEEP_N_PER_EMOTION = 20
JUDGE = True                         # also score expressiveness (unsteered judge)

# %%
token = getpass.getpass("Enter your Hugging Face token: ")
huggingface_hub.login(token=token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# %%
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'  # left-pad for batched decoder-only generation

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
model.eval()
MODEL_DEVICE = next(model.parameters()).device
NUM_DECODER_LAYERS = model.config.num_hidden_layers
print(f"Model loaded. Decoder layers: {NUM_DECODER_LAYERS}")

# %%
def load_context_embeddings_npz(filepath, emotions):
    """Load context_embeddings.npz ('emotion_layer' keys) → {emotion: {layer: vec}}."""
    data = np.load(filepath)
    context_embs = defaultdict(dict)
    for key in data.files:
        context, layer_idx = key.rsplit('_', 1)
        context_embs[context][int(layer_idx)] = data[key]
    return context_embs

context_embs = load_context_embeddings_npz(STEP1_PATH + 'context_embeddings.npz', EMOTIONS)
print(f"Context prototypes loaded for: {list(context_embs.keys())}")

# %%
class LayerSteerer:
    """Adds a steering push to a single decoder layer's residual output.

    mode='norm_matched': hidden += coeff * ||hidden|| * unit_vec  (per token position)
    mode='fixed':        hidden += coeff * unit_vec

    Steering vector / coeff are mutable so the same attached hook can be reused
    across emotions (congruent vec changes per emotion) and disabled for judging.
    """
    def __init__(self, model, mode='norm_matched'):
        self.model = model
        self.mode = mode
        self.unit_vec = None   # (hidden_dim,) float32 tensor on MODEL_DEVICE
        self.coeff = 0.0
        self.handle = None

    def _hook(self, module, inputs, output):
        if self.unit_vec is None or self.coeff == 0.0:
            return output
        # Depending on the transformers version, a decoder layer returns either a
        # bare hidden-states tensor or a tuple (hidden_states, ...). Handle both.
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        uv = self.unit_vec.to(hidden.dtype)
        if self.mode == 'norm_matched':
            # per-position norm in float32 to avoid fp16 overflow on large activations
            norm = hidden.detach().float().norm(dim=-1, keepdim=True)   # (batch, seq, 1)
            hidden = hidden + (self.coeff * norm).to(hidden.dtype) * uv
        else:
            hidden = hidden + self.coeff * uv
        return (hidden,) + tuple(output[1:]) if is_tuple else hidden

    def attach(self, hs_layer):
        """hs_layer in 1..NUM_DECODER_LAYERS (hidden_states index)."""
        assert 1 <= hs_layer <= NUM_DECODER_LAYERS, \
            f"layer {hs_layer} out of range 1..{NUM_DECODER_LAYERS}"
        self.detach()
        block = self.model.model.layers[hs_layer - 1]  # output == hidden_states[hs_layer]
        self.handle = block.register_forward_hook(self._hook)

    def detach(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def set(self, unit_vec, coeff):
        self.unit_vec, self.coeff = unit_vec, coeff

    def disable(self):
        self.unit_vec, self.coeff = None, 0.0

steerer = LayerSteerer(model, mode=STEER_MODE)

def unit_prototype(emotion, hs_layer):
    """Unit-normalized raw context prototype for (emotion, layer), on device."""
    v = torch.tensor(context_embs[emotion][hs_layer], dtype=torch.float32,
                     device=MODEL_DEVICE)
    return v / v.norm().clamp(min=1e-8)

# %%
# ---- Generation setup (mirrors Step-3 zeroshot so outputs stay comparable) ----
GEN_SYSTEM_PROMPT = (
    "You are a medical assistant. Given the user's emotional situation, do two things:\n"
    "1. Write a response (1-2 sentences).\n"
    "2. Rate how emotional is the user's situation on a scale of 1 (mild) to 5 (highly emotional).\n\n"
    "Format your output exactly as:\n"
    "Response: <your response>\n"
    "Intensity: <1-5>"
)

def build_eval_prompt(generated_response):
    return (
        "Rate how emotionally expressive this response is on a scale of 1 (mild) to 5 (highly emotional). "
        "Reply with only a single number.\n\nResponse: " + generated_response
    )

BATCH_SIZE = 8
GEN_MAX_NEW_TOKENS  = 100
EVAL_MAX_NEW_TOKENS = 5

def format_chat(system_content, user_content):
    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def generate_batch(prompts, max_new_tokens, do_sample=True):
    """Batched generation. Steering (if active on `steerer`) applies via the hook."""
    outputs = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i + BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                           truncation=True, max_length=512).to(MODEL_DEVICE)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=0.7 if do_sample else None,
                top_p=0.9 if do_sample else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs['input_ids'].shape[1]
        for out in output_ids:
            outputs.append(tokenizer.decode(out[input_len:], skip_special_tokens=True).strip())
        del inputs, output_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return outputs

def parse_score(text):
    for ch in text:
        if ch in '12345':
            return int(ch)
    return None

def parse_response_intensity(text):
    response, intensity = text, None
    for line in text.split('\n'):
        s = line.strip()
        if s.lower().startswith('response:'):
            response = s[len('response:'):].strip()
        elif s.lower().startswith('intensity:'):
            intensity = parse_score(s[len('intensity:'):])
    return response, intensity

# %%
with open(BASE_PATH + 'prompts_utterances_set2.json') as f:
    data = json.load(f)

def sample_items(emotion):
    items = data[emotion]
    return items if SWEEP_N_PER_EMOTION is None else items[:SWEEP_N_PER_EMOTION]

# %%
def run_config(hs_layer, coeff):
    """Congruent-steer a single layer at strength coeff; generate + (optionally) judge.

    Returns (results_by_emotion, log_row).
    """
    steerer.attach(hs_layer)
    results = {}
    for emotion in EMOTIONS:
        steerer.set(unit_prototype(emotion, hs_layer), coeff)  # congruent
        items      = sample_items(emotion)
        utterances = [it['first_utterance'] for it in items]
        prompts    = [it['prompt'] for it in items]
        chat       = [format_chat(GEN_SYSTEM_PROMPT, u) for u in utterances]
        raw        = generate_batch(chat, GEN_MAX_NEW_TOKENS, do_sample=True)
        parsed     = [parse_response_intensity(r) for r in raw]
        results[emotion] = [
            {"emotion": emotion, "prompt": p, "utterance": u,
             "response": resp, "self_intensity": inten,
             "steer_layer": hs_layer, "coeff": coeff, "steer_mode": STEER_MODE,
             "vector_type": VECTOR_TYPE}
            for p, u, (resp, inten) in zip(prompts, utterances, parsed)
        ]

    # Judge WITHOUT steering (judge should read the text neutrally).
    if JUDGE:
        steerer.disable()
        for emotion in EMOTIONS:
            items = results[emotion]
            eval_prompts = [format_chat(None, build_eval_prompt(it['response'])) for it in items]
            judge_out = generate_batch(eval_prompts, EVAL_MAX_NEW_TOKENS, do_sample=False)
            for it, jo in zip(items, judge_out):
                it['judged_score'] = parse_score(jo)

    steerer.detach()

    flat = [it for items in results.values() for it in items]
    judged = [it['judged_score'] for it in flat if it.get('judged_score') is not None]
    intens = [it['self_intensity'] for it in flat if it['self_intensity'] is not None]
    log_row = {
        'layer_steered': hs_layer,
        'coeff': coeff,
        'steer_mode': STEER_MODE,
        'vector_type': VECTOR_TYPE,
        'token_positions': TOKEN_POSITIONS,
        'n_per_emotion': SWEEP_N_PER_EMOTION,
        'n_total': len(flat),
        'judged_score_mean': round(float(np.mean(judged)), 4) if judged else None,
        'judged_score_parsed': len(judged),
        'self_intensity_mean': round(float(np.mean(intens)), 4) if intens else None,
    }
    return results, log_row

# %%
# ============================ SWEEP ============================
log_path = STEP4_PATH + 'steering_sweep_log.json'
sweep_log = json.load(open(log_path)) if os.path.exists(log_path) else []
def _key(r):
    return (r['layer_steered'], r['coeff'], r['steer_mode'], r['vector_type'])
done = {_key(r) for r in sweep_log}

for hs_layer in STEER_LAYERS:
    for coeff in COEFFS:
        key = (hs_layer, coeff, STEER_MODE, VECTOR_TYPE)
        out_name = f"steered_{VECTOR_TYPE}_{STEER_MODE}_L{hs_layer}_c{coeff}.json"
        out_path = STEP4_PATH + out_name
        if key in done and os.path.exists(out_path):
            print(f"Skip layer {hs_layer} coeff {coeff} (done).")
            continue

        print(f"\n=== Steer layer {hs_layer} | coeff {coeff} | {STEER_MODE} | {VECTOR_TYPE} ===")
        results, log_row = run_config(hs_layer, coeff)
        log_row['output_file'] = out_name

        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        sweep_log = [r for r in sweep_log if _key(r) != key]
        sweep_log.append(log_row)
        with open(log_path, 'w') as f:
            json.dump(sweep_log, f, indent=2)

        print(f"  judged_score_mean={log_row['judged_score_mean']} "
              f"self_intensity_mean={log_row['self_intensity_mean']} | saved {out_name}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print(f"\nDone. Sweep log: {log_path}")
print("Next: embed steered responses + run the probe to add probe_accuracy per config.")
