# %%
"""Build Set 1 (prompts, for probing) and Set 2 (prompt + first utterance, for
generation / steering) from EmpatheticDialogues.

Changes vs the previous version:
  * 'faithful' dropped   - only 443 unique prompts, too few for a disjoint 400/200 split.
  * Set 1 = 400 prompts per emotion (was 200) - more data for the linear probe.
  * Set 1 and Set 2 are drawn WITHOUT overlap, so a probe trained on Set 1 prompts is
    never tested on utterances describing a situation it already saw.
  * '_comma_' dataset artifact is replaced with a real comma.
  * Label-word leakage (a prompt literally containing its own emotion word) is counted
    and reported so it can be accounted for when interpreting probe accuracy.
  * Writes to Llama-3.2-3B-Instruct/Step-6/data/ - no existing file is overwritten.
"""
import json
import os

import pandas as pd

# %%
SEED = 42

# 10 emotions: 'faithful' excluded (see module docstring).
TARGET_CONTEXTS = ['afraid', 'angry', 'anxious', 'devastated', 'lonely', 'sad',
                   'terrified', 'furious', 'grateful', 'hopeful']

N_SET1 = 400   # prompts per emotion -> probing
N_SET2 = 200   # prompt+utterance pairs per emotion -> generation / steering
# Smallest available pool is 'devastated' at 665 unique prompts, so 400+200=600 fits all 10.

BASE_PATH = '/Volumes/Reading/Projects/Empathy-LLM/'
# BASE_PATH = '/content/drive/MyDrive/Project-4/'

ED_DIR = BASE_PATH + 'empatheticdialogues/'                    # raw EmpatheticDialogues CSVs
OUT_DIR = BASE_PATH + 'Llama-3.2-3B-Instruct/Step-6/data/'     # all Step-6 artifacts
os.makedirs(OUT_DIR, exist_ok=True)

SPLITS = ['train', 'test', 'valid']


# %%
def clean(text):
    """Undo the EmpatheticDialogues '_comma_' escape and normalise whitespace."""
    return ' '.join(str(text).replace('_comma_', ',').split())


# %%
def load_unique_prompts():
    """All splits combined, filtered to TARGET_CONTEXTS, one row per unique prompt.

    For each prompt the utterance with the lowest utterance_idx is kept: that is the
    speaker's own first turn, i.e. the text the model will later be asked to respond to.
    """
    frames = []
    for split in SPLITS:
        df = pd.read_csv(ED_DIR + f'{split}.csv',
                         usecols=['context', 'prompt', 'utterance', 'utterance_idx'],
                         on_bad_lines='skip')
        frames.append(df[df['context'].isin(TARGET_CONTEXTS)])

    df = pd.concat(frames, ignore_index=True)
    df['utterance_idx'] = pd.to_numeric(df['utterance_idx'], errors='coerce')
    df['prompt'] = df['prompt'].map(clean)
    df['utterance'] = df['utterance'].map(clean)

    return (df.sort_values('utterance_idx')
              .drop_duplicates(subset=['context', 'prompt'], keep='first')
              [['context', 'prompt', 'utterance']]
              .reset_index(drop=True))


df_unique = load_unique_prompts()
pool_sizes = df_unique.groupby('context')['prompt'].nunique().reindex(TARGET_CONTEXTS)
print('Unique prompts available per emotion:')
print(pool_sizes.to_string())

need = N_SET1 + N_SET2
short = pool_sizes[pool_sizes < need]
assert short.empty, f'Not enough unique prompts for {need} per emotion: {short.to_dict()}'
print(f'\nAll emotions have >= {need} unique prompts. Proceeding.')

# %%
# Disjoint split: sample N_SET1+N_SET2 rows per emotion, first block -> Set 1, rest -> Set 2.
set1, set2, rows_set1, rows_set2 = {}, {}, [], []

for ctx in TARGET_CONTEXTS:
    pool = df_unique[df_unique['context'] == ctx]
    sampled = pool.sample(n=need, random_state=SEED).reset_index(drop=True)

    s1, s2 = sampled.iloc[:N_SET1], sampled.iloc[N_SET1:]
    assert set(s1['prompt']).isdisjoint(set(s2['prompt']))

    set1[ctx] = s1['prompt'].tolist()
    set2[ctx] = [{'prompt': r.prompt, 'first_utterance': r.utterance}
                 for r in s2.itertuples()]

    rows_set1.extend({'context': ctx, 'prompt': p} for p in set1[ctx])
    rows_set2.extend({'context': ctx, **e} for e in set2[ctx])

print('Set 1 (prompts):    ', {c: len(v) for c, v in set1.items()})
print('Set 2 (prompt+utt): ', {c: len(v) for c, v in set2.items()})

# %%
# Label-word leakage: prompts that literally contain their own emotion word are trivially
# classifiable from surface form, so they inflate probe accuracy. Counted, not removed -
# removing them would bias the sample; the count lets us caveat the result instead.
leakage = {ctx: sum(1 for p in set1[ctx] if ctx in p.lower()) for ctx in TARGET_CONTEXTS}
print('\nSet 1 prompts containing their own emotion word:')
for ctx, n in leakage.items():
    print(f'  {ctx:12s} {n:3d} / {N_SET1}  ({n / N_SET1:.1%})')

# %%
set1_path = OUT_DIR + 'prompts_set1.json'
set2_path = OUT_DIR + 'prompts_utterances_set2.json'

with open(set1_path, 'w') as f:
    json.dump(set1, f, indent=2)
with open(set2_path, 'w') as f:
    json.dump(set2, f, indent=2)

# Flat CSVs for eyeballing the data.
pd.DataFrame(rows_set1).to_csv(OUT_DIR + 'prompts_set1.csv', index=False)
pd.DataFrame(rows_set2).to_csv(OUT_DIR + 'prompts_utterances_set2.csv', index=False)

stats = {
    'emotions': TARGET_CONTEXTS,
    'dropped_emotions': ['faithful'],
    'n_set1_per_emotion': N_SET1,
    'n_set2_per_emotion': N_SET2,
    'seed': SEED,
    'unique_prompt_pool': {c: int(n) for c, n in pool_sizes.items()},
    'set1_label_word_leakage': leakage,
    'chance_accuracy': 1.0 / len(TARGET_CONTEXTS),
}
with open(OUT_DIR + 'dataset_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f'\nWrote:\n  {set1_path}\n  {set2_path}\n  {OUT_DIR}dataset_stats.json')
