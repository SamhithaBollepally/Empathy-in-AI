
#%%
# from google.colab import drive
# drive.mount('/content/drive')

# %%
import os
import json
import gc
import getpass
import torch
import huggingface_hub
from transformers import AutoTokenizer, AutoModelForCausalLM

# %%
BASE_PATH = '/Volumes/Reading/Projects/Empathy-LLM/Llama-3.2-3B-Instruct/'
#BASE_PATH = '/content/drive/MyDrive/Project-4/'

EMOTIONS = ['afraid', 'angry', 'anxious', 'devastated', 'lonely', 'sad',
            'terrified', 'furious', 'grateful', 'hopeful', 'faithful']

STEP3_PATH = BASE_PATH + 'Step-3/'
os.makedirs(STEP3_PATH, exist_ok=True)

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
tokenizer.padding_side = 'left'     # left-pad for batched decoder-only generation
tokenizer.truncation_side = 'left'  # keep the trailing assistant cue if truncation triggers

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
model.eval()
MODEL_DEVICE = next(model.parameters()).device  # compute once
print("Model loaded.")

# %%
# ---- Stage 1: generation prompt (produces the empathetic response only) ----
GEN_SYSTEM_PROMPT = (
    "You are a medical assistant. Before responding, reason through:\n"
    "1. What emotion is this person experiencing?\n"
    "2. Why might they feel this way?\n"
    "3. What tone or approach would be most helpful?\n"
    "4. How emotional is the user's situation on a scale of 1 (mild) to 5 (highly emotional).\n\n"
    "Then provide your output in exactly this format:\n"
    "Reasoning: <your step-by-step thinking>\n"
    "Response: <your response (1-2 sentences)>\n"
    "Intensity: <1-5>"
)
# ---- Stage 2: judge prompt (scores a generated response's expressiveness) ----
def build_eval_prompt(generated_response):
    return (
        "Rate how emotionally expressive this response is on a scale of 1 (mild) to 5 (highly emotional). "
        "Reply with only a single number.\n\nResponse: " + generated_response
    )

BATCH_SIZE = 8
GEN_MAX_NEW_TOKENS  = 512   # room for reasoning + response + intensity without truncation
EVAL_MAX_NEW_TOKENS = 5    

def format_chat(system_content, user_content):
    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def generate_batch(prompts, max_new_tokens, do_sample=True):
    """Batched generation over a list of pre-formatted chat prompts."""
    outputs = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=False  # chat template already added BOS/header tokens
        ).to(MODEL_DEVICE)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=0.7 if do_sample else None,
                top_p=0.9 if do_sample else None,
                pad_token_id=tokenizer.eos_token_id
            )

        input_len = inputs['input_ids'].shape[1]
        for out in output_ids:
            text = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            outputs.append(text.strip())

        del inputs, output_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return outputs

def parse_score(text):
    """Extract the first digit 1-5 from the judge output; None if not found."""
    for ch in text:
        if ch in '12345':
            return int(ch)
    return None

def _label_remainder(line, label):
    """If `line` starts with `label` (case-insensitive, ignoring leading/trailing
    markdown emphasis like ** or #), return the text after the label; else None.
    """
    norm = line.lstrip('*#->  \t').strip()  # tolerate '**Response:**', '- Response:', etc.
    if norm.lower().startswith(label):
        return norm[len(label):].strip().strip('*').strip()
    return None

def parse_cot_output(text):
    """Split Stage 1 output into (reasoning_str, response_str, intensity_int, format_ok).

    Both reasoning and response may span multiple lines:
      - reasoning = everything between 'Reasoning:' and the next label
      - response  = everything between 'Response:' and 'Intensity:' (or end of text)
    Robust to markdown-wrapped labels ('**Response:**'). format_ok is True only when a
    non-empty response was captured, so malformed/empty output is NOT passed downstream.
    """
    reasoning_parts, response_parts = [], []
    intensity = None
    state = None  # None | 'reasoning' | 'response'

    for line in text.split('\n'):
        stripped = line.strip()
        rest = _label_remainder(stripped, 'reasoning:')
        if rest is not None:
            state = 'reasoning'
            if rest:
                reasoning_parts.append(rest)
            continue
        rest = _label_remainder(stripped, 'response:')
        if rest is not None:
            state = 'response'
            if rest:
                response_parts.append(rest)
            continue
        rest = _label_remainder(stripped, 'intensity:')
        if rest is not None:
            state = None
            intensity = parse_score(rest)
            continue
        if state == 'reasoning':
            reasoning_parts.append(stripped)
        elif state == 'response':
            response_parts.append(stripped)

    reasoning = '\n'.join(reasoning_parts).strip()
    response = '\n'.join(response_parts).strip()
    format_ok = bool(response)
    return reasoning, response, intensity, format_ok

# %%
with open(BASE_PATH + 'prompts_utterances_set2.json') as f:
    data = json.load(f)

output_path = STEP3_PATH + 'CoT_responses.json'

# %%
# ============================ STAGE 1: GENERATE ============================
# Load existing checkpoint if run was interrupted
if os.path.exists(output_path):
    with open(output_path) as f:
        results = json.load(f)
    print(f"Resuming from checkpoint: {list(results.keys())} already done.")
else:
    results = {}

for emotion in EMOTIONS:
    if emotion in results:
        print(f"Skipping {emotion} (already generated).")
        continue
    if emotion not in data:
        print(f"Skipping {emotion} (not present in input data).")
        continue
    print(f"\n[Stage 1] Generating responses for: {emotion} ({len(data[emotion])} utterances)...")
    prompt_texts = [item['prompt']          for item in data[emotion]]
    utterances   = [item['first_utterance'] for item in data[emotion]]
    chat_prompts = [format_chat(GEN_SYSTEM_PROMPT, u) for u in utterances]
    raw_out      = generate_batch(chat_prompts, GEN_MAX_NEW_TOKENS, do_sample=True)
    parsed       = [parse_cot_output(r) for r in raw_out]
    results[emotion] = [
        {"emotion": emotion, "prompt": p, "utterance": u,
         "reasoning": reason, "response": resp, "self_intensity": inten,
         "format_ok": ok}
        for p, u, (reason, resp, inten, ok) in zip(prompt_texts, utterances, parsed)
    ]
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    n_parsed = sum(1 for it in results[emotion] if it['self_intensity'] is not None)
    print(f"  Saved checkpoint ({len(results)}/{len(EMOTIONS)} emotions done) "
          f"| self_intensity parsed: {n_parsed}/{len(raw_out)}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\n[Stage 1] Done. Responses saved to: {output_path}")

# %%
# ======================= BACKFILL EMPTY RESPONSES ========================
# Regenerates ONLY entries whose 'response' is empty (e.g. produced before the
# multi-line parser fix). Existing non-empty responses are left untouched.
# Re-runnable: keeps regenerating until no empties remain (or gives up per-run).
with open(output_path) as f:
    results = json.load(f)

for emotion in EMOTIONS:
    if emotion not in results:
        continue
    items = results[emotion]
    todo = [(i, it) for i, it in enumerate(items) if not (it.get('response') or '').strip()]
    if not todo:
        continue
    print(f"\n[Backfill] {emotion}: regenerating {len(todo)} empty responses...")
    chat_prompts = [format_chat(GEN_SYSTEM_PROMPT, it['utterance']) for _, it in todo]
    raw_out      = generate_batch(chat_prompts, GEN_MAX_NEW_TOKENS, do_sample=True)
    parsed       = [parse_cot_output(r) for r in raw_out]
    for (idx, _), (reason, resp, inten, ok) in zip(todo, parsed):
        items[idx].update({
            "reasoning": reason, "response": resp,
            "self_intensity": inten, "format_ok": ok,
        })
        # A backfilled response invalidates any stale judge score.
        items[idx].pop('judged_score', None)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    recovered = sum(1 for (idx, _) in todo if (items[idx].get('response') or '').strip())
    print(f"  Recovered {recovered}/{len(todo)} (still empty: {len(todo) - recovered})")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

remaining = sum(1 for items in results.values()
                for it in items if not (it.get('response') or '').strip())
print(f"\n[Backfill] Done. Remaining empty responses: {remaining}")

# %%
# ============================ STAGE 2: JUDGE ==============================
# Re-runnable independently — reads the saved responses and adds 'judged_score'.
with open(output_path) as f:
    results = json.load(f)

for emotion in EMOTIONS:
    if emotion not in results:
        print(f"Skipping {emotion} (no Stage 1 output).")
        continue
    items = results[emotion]
    if all('judged_score' in it for it in items):
        print(f"Skipping {emotion} (already judged).")
        continue
    print(f"\n[Stage 2] Judging responses for: {emotion} ({len(items)} responses)...")
    # Only judge well-formed, non-empty responses; malformed ones get judged_score=None.
    judgeable = [it for it in items if it.get('response')]
    for it in items:
        if not it.get('response'):
            it['judged_score'] = None
    eval_prompts = [format_chat(None, build_eval_prompt(it['response'])) for it in judgeable]
    judge_out    = generate_batch(eval_prompts, EVAL_MAX_NEW_TOKENS, do_sample=False)
    for it, jo in zip(judgeable, judge_out):
        it['judged_score'] = parse_score(jo)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    n_scored = sum(1 for it in items if it['judged_score'] is not None)
    print(f"  Saved. Scores parsed: {n_scored}/{len(items)} "
          f"(skipped {len(items) - len(judgeable)} malformed)")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# %%
print(f"\nSaved: {output_path}")
print(f"Total responses: {sum(len(v) for v in results.values())}")

# %%
# ==================== SPLIT INTO TWO OUTPUT FILES ====================
# File 1: utterances + reasoning
# File 2: emotion, prompt, utterance, response, intensity, judgement
with open(output_path) as f:
    results = json.load(f)

reasoning_path = STEP3_PATH + 'CoT_reasoning_utterances.json'
scores_path    = STEP3_PATH + 'CoT_responses_scores.json'

reasoning_only = {
    emotion: [
        {"emotion": emotion, "utterance": it["utterance"], "reasoning": it.get("reasoning", "")}
        for it in items
    ]
    for emotion, items in results.items()
}
scores_only = {
    emotion: [
        {"emotion": emotion, "prompt": it["prompt"], "utterance": it["utterance"],
         "response": it["response"], "intensity": it.get("self_intensity"),
         "judgement": it.get("judged_score")}
        for it in items
    ]
    for emotion, items in results.items()
}

with open(reasoning_path, 'w') as f:
    json.dump(reasoning_only, f, indent=2)
with open(scores_path, 'w') as f:
    json.dump(scores_only, f, indent=2)

n_total = sum(len(v) for v in results.values())
print(f"Saved reasoning + utterances to: {reasoning_path}")
print(f"Saved responses + scores  to:    {scores_path}")
print(f"Entries per file: {n_total}")
