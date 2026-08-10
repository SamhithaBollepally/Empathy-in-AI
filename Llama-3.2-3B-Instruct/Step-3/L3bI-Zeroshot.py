
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
tokenizer.padding_side = 'left'  # left-pad for batched decoder-only generation

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
    "You are a medical assistant. Given the user's emotional situation, do two things:\n"
    "1. Write a response (1-2 sentences).\n"
    "2. Rate how emotional is the user's situation on a scale of 1 (mild) to 5 (highly emotional).\n\n"
    "Format your output exactly as:\n"
    "Response: <your response>\n"
    "Intensity: <1-5>"
)

# ---- Stage 2: judge prompt (scores a generated response's expressiveness) ----
def build_eval_prompt(generated_response):
    return (
        "Rate how emotionally expressive this response is on a scale of 1 (mild) to 5 (highly emotional). "
        "Reply with only a single number.\n\nResponse: " + generated_response
    )

BATCH_SIZE = 8
GEN_MAX_NEW_TOKENS  = 100   # 1-2 sentence response
EVAL_MAX_NEW_TOKENS = 5    # a single digit score

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
            max_length=512
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

def parse_response_intensity(text):
    """Split Stage 1 output 'Response: ...\nIntensity: X' into (response_str, intensity_int).
    Falls back to the raw text as response and None intensity if the format isn't followed.
    """
    response, intensity = text, None
    for line in text.split('\n'):
        stripped = line.strip()
        if stripped.lower().startswith('response:'):
            response = stripped[len('response:'):].strip()
        elif stripped.lower().startswith('intensity:'):
            intensity = parse_score(stripped[len('intensity:'):])
    return response, intensity

# %%
with open(BASE_PATH + 'prompts_utterances_set2.json') as f:
    data = json.load(f)

output_path = STEP3_PATH + 'zeroshot_responses.json'

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
    print(f"\n[Stage 1] Generating responses for: {emotion} ({len(data[emotion])} utterances)...")
    prompt_texts = [item['prompt']          for item in data[emotion]]
    utterances   = [item['first_utterance'] for item in data[emotion]]
    chat_prompts = [format_chat(GEN_SYSTEM_PROMPT, u) for u in utterances]
    raw_out      = generate_batch(chat_prompts, GEN_MAX_NEW_TOKENS, do_sample=True)
    parsed       = [parse_response_intensity(r) for r in raw_out]
    results[emotion] = [
        {"emotion": emotion, "prompt": p, "utterance": u,
         "response": resp, "self_intensity": inten}
        for p, u, (resp, inten) in zip(prompt_texts, utterances, parsed)
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
# ============================ STAGE 2: JUDGE ==============================
# Re-runnable independently — reads the saved responses and adds 'judged_score'.
with open(output_path) as f:
    results = json.load(f)

for emotion in EMOTIONS:
    items = results[emotion]
    if all('judged_score' in it for it in items):
        print(f"Skipping {emotion} (already judged).")
        continue
    print(f"\n[Stage 2] Judging responses for: {emotion} ({len(items)} responses)...")
    eval_prompts = [format_chat(None, build_eval_prompt(it['response'])) for it in items]
    judge_out    = generate_batch(eval_prompts, EVAL_MAX_NEW_TOKENS, do_sample=False)
    for it, jo in zip(items, judge_out):
        it['judged_score'] = parse_score(jo)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    n_scored = sum(1 for it in items if it['judged_score'] is not None)
    print(f"  Saved. Scores parsed: {n_scored}/{len(items)}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# %%
print(f"\nSaved: {output_path}")
print(f"Total responses: {sum(len(v) for v in results.values())}")
