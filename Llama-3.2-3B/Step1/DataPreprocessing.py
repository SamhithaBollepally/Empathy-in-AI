import os
import pandas as pd

# Define the folder path
folder_path = 'empatheticdialogues'

# Get all files in the folder
files = os.listdir(folder_path)

# Read all CSV files
data = {}
for file in files:
    if file.endswith('.csv'):
        file_path = os.path.join(folder_path, file)
        data[file] = pd.read_csv(file_path, on_bad_lines='skip')
        print(f"Loaded {file} with shape: {data[file].shape}")

print(f"\nTotal files loaded: {len(data)}")

for file, df in data.items():
    print(f"\nColumns in {file}: {df.columns.tolist()}")

# # Example: Filter rows where context == 'anxious'
# print("\n--- Example: Filtering for context == 'anxious' ---")
# example_df = data['train.csv'][data['train.csv']['context'] == 'anxious']
# print(f"Number of rows with context 'anxious' in train.csv: {len(example_df)}")
# print(f"First few rows:\n{example_df.head()}")

# Define the contexts to filter
target_contexts = ['afraid', 'angry', 'anxious', 'devastated', 'lonely', 'sad', 'terrified', 'furious', 'grateful', 'hopeful', 'faithful']

# Filter train.csv and test.csv for target contexts
train_filtered = data['train.csv'][data['train.csv']['context'].isin(target_contexts)]
test_filtered = data['test.csv'][data['test.csv']['context'].isin(target_contexts)]

print(f"\nFiltered train.csv: {len(train_filtered)} rows")
print(f"Filtered test.csv: {len(test_filtered)} rows")

# Combine train and test filtered data into a single file
combined_filtered = pd.concat([train_filtered, test_filtered], ignore_index=True)
print(f"Combined filtered data: {len(combined_filtered)} rows")

# Save to a single CSV file
combined_filtered.to_csv('combined_filtered.csv', index=False)

print("\nSaved combined filtered file: combined_filtered.csv")

# Count unique prompts for each context
print("\n--- Number of unique prompts for each context ---")
for context in target_contexts:
    context_prompts = combined_filtered[combined_filtered['context'] == context]['prompt']
    unique_count = context_prompts.nunique()
    total_count = len(context_prompts)
    print(f"{context}: {unique_count} unique prompts (out of {total_count} total rows)")

# Sample 100 random unique prompts from each context and get all their rows
print("\n--- Sampling 100 random unique prompts from each context ---")
sampled_rows = []

for context in target_contexts:
    context_data = combined_filtered[combined_filtered['context'] == context]
    unique_prompts = context_data['prompt'].unique()
    
    # Sample 100 random unique prompts (or all if less than 100)
    sample_size = min(100, len(unique_prompts))
    sampled_prompts = pd.Series(unique_prompts).sample(n=sample_size, random_state=42).tolist()
    
    # Get all rows for the sampled prompts
    context_sampled = context_data[context_data['prompt'].isin(sampled_prompts)]
    sampled_rows.append(context_sampled)
    
    print(f"{context}: Sampled {sample_size} unique prompts ({len(context_sampled)} total rows)")

# Combine all sampled rows
final_sampled = pd.concat(sampled_rows, ignore_index=True)
print(f"\nTotal sampled rows: {len(final_sampled)}")

# Save to new CSV file
final_sampled.to_csv('sampled_100_per_context.csv', index=False)
print("Saved sampled data to: sampled_100_per_context.csv")

import json

# Load the sampled data
sampled_data = pd.read_csv('sampled_100_per_context.csv')

# Create a dictionary: context -> list of 100 unique prompts
context_prompts_dict = {}

for context in sampled_data['context'].unique():
    context_data = sampled_data[sampled_data['context'] == context]
    unique_prompts = context_data['prompt'].unique().tolist()[:100]  # Ensure exactly 100
    context_prompts_dict[context] = unique_prompts

print(f"Created dictionary with {len(context_prompts_dict)} contexts")
for context, prompts in context_prompts_dict.items():
    print(f"{context}: {len(prompts)} unique prompts")

# Save to JSON (memory-efficient, human-readable, easy to load)
with open('context_prompts.json', 'w') as f:
    json.dump(context_prompts_dict, f, indent=2)

print("\nSaved to context_prompts.json")
