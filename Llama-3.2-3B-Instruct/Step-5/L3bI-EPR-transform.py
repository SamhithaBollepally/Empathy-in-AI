# %%
# ============================================================================
# Step 5i — Inter-Layer Hidden-State Transformation
#
# No injection, no prototypes, no probes.
# Simply transform the output of one layer before feeding it to the next:
#
#   Layer 1 output → transform(elementwise) → Layer 2 input
#
# Modes:
#   'tanh'    → tanh(x)       maps every value to (-1, 1)
#   'square'  → x ** 2        makes every value positive
#   'cubic'   → x + α x^3     sign-preserving, amplifies large values
#   'identity'→ x             baseline, no change
# ============================================================================

# %%
import os
import gc
import json
import getpass
import numpy as np
import torch
import huggingface_hub
from transformers import AutoTokenizer, AutoModelForCausalLM

# %%
# ── Config ───────────────────────────────────────────────────────────────────
BASE_PATH = '/Volumes/Reading/Projects/Empathy-LLM/Llama-3.2-3B-Instruct/'
# BASE_PATH = '/content/drive/MyDrive/Project-4/'

STEP5_PATH = BASE_PATH + 'Step-5/'
os.makedirs(STEP5_PATH, exist_ok=True)

EMOTIONS = ['afraid', 'angry', 'anxious', 'devastated', 'lonely', 'sad',
            'terrified', 'furious', 'grateful', 'hopeful', 'faithful']

STEER_LAYERS     = [26, 27, 28] # layers to transform
TRANSFORM_MODE   = 'cubic'      # 'tanh' | 'square' | 'cubic' | 'identity'
CUBIC_ALPHA      = 1e-2         # coefficient; safe because transform is norm-normalised
N_ITEMS          = 50           # utterances per emotion (None = all)
BATCH_SIZE       = 32           # A100 80GB can handle 32
GEN_MAX_NEW_TOKENS  = 60
EVAL_MAX_NEW_TOKENS = 5

# %%
def transform_hidden(hidden, mode):
    """Elementwise transformation of the hidden state tensor."""
    if mode == 'tanh':
        return torch.tanh(hidden)
    elif mode == 'square':
        return hidden * hidden
    elif mode == 'cubic':
        # Norm-normalised cubic residual: x + alpha * (x/norm)^3 * norm
        # Normalising by the token norm prevents the cubic term from exploding
        # at late layers where hidden-state norms are large.
        x    = hidden.float()
        norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_n  = x / norm                               # unit-normed
        result = x + CUBIC_ALPHA * norm * (x_n * x_n * x_n)   # bounded cubic
        result = torch.nan_to_num(result, nan=0.0, posinf=65504.0, neginf=-65504.0)
        result = torch.clamp(result, min=-65504.0, max=65504.0)
        return result.to(hidden.dtype)
    elif mode == 'identity':
        return hidden
    else:
        raise ValueError(f"Unknown TRANSFORM_MODE: {mode}")

# %%
# ── Load model ───────────────────────────────────────────────────────────────
token = getpass.getpass("HuggingFace token: ")
huggingface_hub.login(token=token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = 'left'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
)
model.eval()
MODEL_DEVICE = next(model.parameters()).device
print(f"Model loaded on {MODEL_DEVICE}")

# %%
# ── Transform hook ───────────────────────────────────────────────────────────
class TransformHook:
    def __init__(self, model, steer_layers, mode):
        self.model = model
        self.steer_layers = steer_layers
        self.mode = mode
        self.active = True
        self.handles = []

    def _make_hook(self):
        def hook(module, inputs, output):
            if not self.active:
                return output
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output   # (batch, seq, 3072)
            transformed = transform_hidden(hidden, self.mode)
            return (transformed,) + tuple(output[1:]) if is_tuple else transformed
        return hook

    def attach(self):
        for L in self.steer_layers:
            h = self.model.model.layers[L - 1].register_forward_hook(self._make_hook())
            self.handles.append(h)
        print(f"Transform hooks attached after layers {self.steer_layers} mode={self.mode}")

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def disable(self):
        self.active = False

    def enable(self):
        self.active = True

# %%
# ── Generation helpers ───────────────────────────────────────────────────────
GEN_SYSTEM_PROMPT = (
    "You are a medical assistant. Given the user's emotional situation, do two things:\n"
    "1. Write a response (1-2 sentences).\n"
    "2. Rate how emotional is the user's situation on a scale of 1 (mild) to 5 (highly emotional).\n\n"
    "Format your output exactly as:\n"
    "Response: <your response>\n"
    "Intensity: <1-5>"
)

def build_eval_prompt(response):
    return (
        "Rate how emotionally expressive this response is on a scale of 1 (mild) to 5 (highly emotional). "
        "Reply with only a single number.\n\nResponse: " + response
    )

def format_chat(system, user):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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
# ── Generate with transformed hidden states ──────────────────────────────────
with open(BASE_PATH + 'prompts_utterances_set2.json') as f:
    data = json.load(f)

thook = TransformHook(model, STEER_LAYERS, TRANSFORM_MODE)
thook.attach()

if TRANSFORM_MODE == 'cubic':
    out_path = STEP5_PATH + f'epr_transform_{TRANSFORM_MODE}_a{CUBIC_ALPHA}_l{STEER_LAYERS}.json'
    log_path = STEP5_PATH + f'epr_transform_{TRANSFORM_MODE}_a{CUBIC_ALPHA}_log.json'
else:
    out_path = STEP5_PATH + f'epr_transform_{TRANSFORM_MODE}_l{STEER_LAYERS}.json'
    log_path = STEP5_PATH + f'epr_transform_{TRANSFORM_MODE}_log.json'
out_path = out_path.replace('[', '').replace(']', '').replace(' ', '')
log_path = log_path.replace('[', '').replace(']', '').replace(' ', '')

if os.path.exists(out_path):
    with open(out_path) as f:
        results = json.load(f)
    print(f"Resuming from {list(results.keys())}")
else:
    results = {}

for emotion in EMOTIONS:
    if emotion in results:
        print(f"Skipping {emotion} (checkpoint found).")
        continue

    items = data[emotion] if N_ITEMS is None else data[emotion][:N_ITEMS]
    print(f"\n── {emotion} ({len(items)} utterances) ──")
    results[emotion] = []

    for i in range(0, len(items), BATCH_SIZE):
        batch_items = items[i:i + BATCH_SIZE]
        prompts_    = [it['prompt']          for it in batch_items]
        utterances  = [it['first_utterance'] for it in batch_items]

        thook.enable()
        chats = [format_chat(GEN_SYSTEM_PROMPT, u) for u in utterances]
        inputs = tokenizer(chats, return_tensors='pt', padding=True,
                           truncation=True, max_length=512).to(MODEL_DEVICE)
        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs['input_ids'].shape[1]
        raw = [tokenizer.decode(ids[input_len:], skip_special_tokens=True).strip()
               for ids in out_ids]
        parsed = [parse_response_intensity(r) for r in raw]

        # Judge without transform
        thook.disable()
        eval_chats = [format_chat(None, build_eval_prompt(resp)) for resp, _ in parsed]
        eval_inputs = tokenizer(eval_chats, return_tensors='pt', padding=True,
                                truncation=True, max_length=512).to(MODEL_DEVICE)
        with torch.inference_mode():
            judge_ids = model.generate(
                **eval_inputs,
                max_new_tokens=EVAL_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        j_len = eval_inputs['input_ids'].shape[1]
        judge_out = [tokenizer.decode(ids[j_len:], skip_special_tokens=True).strip()
                     for ids in judge_ids]

        for p, u, (resp, inten), jo in zip(prompts_, utterances, parsed, judge_out):
            results[emotion].append({
                "emotion":        emotion,
                "prompt":         p,
                "utterance":      u,
                "response":       resp,
                "self_intensity": inten,
                "judged_score":   parse_score(jo),
                "transform_mode": TRANSFORM_MODE,
                "steer_layers":   STEER_LAYERS,
                "method":         "epr_transform",
            })

        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

        del inputs, out_ids, eval_inputs, judge_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    judged = [it['judged_score']   for it in results[emotion] if it['judged_score']   is not None]
    intens = [it['self_intensity'] for it in results[emotion] if it['self_intensity'] is not None]
    print(f"  judged_mean={round(float(np.mean(judged)), 4) if judged else None}  "
          f"intensity_mean={round(float(np.mean(intens)), 4) if intens else None}")

thook.detach()

# %%
# ── Save summary log ─────────────────────────────────────────────────────────
per_emotion = []
for e in EMOTIONS:
    items = results.get(e, [])
    judged = [it['judged_score']   for it in items if it.get('judged_score')   is not None]
    intens = [it['self_intensity'] for it in items if it.get('self_intensity') is not None]
    per_emotion.append({
        'emotion':             e,
        'n':                   len(items),
        'judged_score_mean':   round(float(np.mean(judged)), 4) if judged else None,
        'self_intensity_mean': round(float(np.mean(intens)), 4) if intens else None,
    })

all_judged = [r['judged_score_mean'] for r in per_emotion if r['judged_score_mean'] is not None]
log = {
    'method':              'epr_transform',
    'transform_mode':      TRANSFORM_MODE,
    'steer_layers':        STEER_LAYERS,
    'output_file':         os.path.basename(out_path),
    'overall_judged_mean': round(float(np.mean(all_judged)), 4) if all_judged else None,
    'per_emotion':         per_emotion,
}
with open(log_path, 'w') as f:
    json.dump(log, f, indent=2)

# %%
print(f"\nResults: {out_path}")
print(f"Log:     {log_path}")
print(f"Overall judged mean: {log['overall_judged_mean']}")
