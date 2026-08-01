# %%
import pandas as pd
import json
import random

random.seed(42)

target_contexts = ['afraid', 'angry', 'anxious', 'devastated', 'lonely', 'sad',
                   'terrified', 'furious', 'grateful', 'hopeful', 'faithful']

# %%
files = {
    'train': '/Volumes/Reading/Projects/Empathy-LLM/empatheticdialogues/train.csv',
    'test':  '/Volumes/Reading/Projects/Empathy-LLM/empatheticdialogues/test.csv',
    'valid': '/Volumes/Reading/Projects/Empathy-LLM/empatheticdialogues/valid.csv',
}
# %%
Prompts = []
Utterances = []
Contexts = []
for split, path in files.items():
    df = pd.read_csv(path, usecols=['context', 'prompt', 'utterance', 'utterance_idx'])
    df = df[df['context'].isin(target_contexts)]
    summary = (
        df.groupby('context')
          .agg(
              context=('context', 'first'),
              num_prompts=('prompt', 'nunique'),
              num_utterances=('utterance', 'count')
          )
          .reindex(target_contexts)
    )
    Prompts.extend(df['prompt'].tolist())
    Utterances.extend(df['utterance'].tolist())
    Contexts.extend(df['context'].tolist())
# %%
print("Total rows (all splits):", len(Prompts))
print("Unique contexts:", len(set(Contexts)))
print("Unique prompts:", len(set(Prompts)))
print("Total utterances:", len(Utterances))

# %%
df_all = pd.DataFrame({'context': Contexts, 'prompt': Prompts})
unique_prompts_per_context = df_all.groupby('context')['prompt'].nunique().reindex(target_contexts)
print("\nUnique prompts per context (all splits):")
print(unique_prompts_per_context.to_string())

# %%
# Re-read combined data with utterance_idx for JSON generation
all_dfs = []
for path in files.values():
    df = pd.read_csv(path, usecols=['context', 'prompt', 'utterance', 'utterance_idx'])
    all_dfs.append(df[df['context'].isin(target_contexts)])
df_combined = pd.concat(all_dfs, ignore_index=True)
df_combined['utterance_idx'] = pd.to_numeric(df_combined['utterance_idx'], errors='coerce')

# %%
# One row per unique prompt: keep the utterance with the lowest utterance_idx
df_first = (
    df_combined
    .sort_values('utterance_idx')
    .drop_duplicates(subset=['context', 'prompt'], keep='first')
    [['context', 'prompt', 'utterance']]
    .reset_index(drop=True)
)

# %%
set1 = {}
set2 = {}

for ctx in target_contexts:
    pool = df_first[df_first['context'] == ctx].reset_index(drop=True)
    sampled = pool.sample(n=400, random_state=42)
    s1 = sampled.iloc[:200]
    s2 = sampled.iloc[200:]
    set1[ctx] = s1['prompt'].tolist()
    set2[ctx] = [
        {'prompt': row['prompt'], 'first_utterance': row['utterance']}
        for _, row in s2.iterrows()
    ]

# %%
set1_path = '/Volumes/Reading/Projects/Empathy-LLM/Llama-3.2-3B-Instruct/prompts_set1.json'
set2_path = '/Volumes/Reading/Projects/Empathy-LLM/Llama-3.2-3B-Instruct/prompts_utterances_set2.json'

with open(set1_path, 'w') as f:
    json.dump(set1, f, indent=2)

with open(set2_path, 'w') as f:
    json.dump(set2, f, indent=2)

print("\nprompts_set1.json created:", {ctx: len(v) for ctx, v in set1.items()})
print("prompts_utterances_set2.json created:", {ctx: len(v) for ctx, v in set2.items()})