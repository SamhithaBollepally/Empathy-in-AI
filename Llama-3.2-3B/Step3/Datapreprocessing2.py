import pandas as pd

# Target contexts
CONTEXTS = ['afraid', 'angry', 'anxious', 'devastated', 'lonely', 'sad', 
            'terrified', 'furious', 'grateful', 'hopeful', 'faithful']

# Load and prepare data
sampled_df = pd.read_csv('Step1/sampled_100_per_context.csv')
excluded_prompts = set(sampled_df['prompt'].str.strip())

full_df = pd.read_csv('Step1/combined_filtered.csv')

# Filter: first utterances only, exclude already sampled prompts
df = full_df[full_df['utterance_idx'] == 1].copy()
df['prompt'] = df['prompt'].str.strip()
df = df[~df['prompt'].isin(excluded_prompts)]

# Sample 100 unique prompts per context
results = []
for context in CONTEXTS:
    context_data = df[df['context'] == context].drop_duplicates(subset=['prompt']).head(100)
    results.append(context_data)

# Create output
output = pd.concat(results, ignore_index=True)[['context', 'prompt', 'utterance']]
output.columns = ['context', 'prompt', 'first_utterance']
output['first_utterance'] = output['first_utterance'].str.strip()

# Save
output.to_csv('Step3/response_prompts.csv', index=False)

# Summary
print(f"Total: {len(output)} prompts")
print("\nPer context:")
for context in CONTEXTS:
    print(f"  {context:<12}: {len(output[output['context'] == context])}")
print("\n✓ Saved to Step3/response_prompts.csv")
