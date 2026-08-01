# Add this cell to the notebook AFTER extracting utterance embeddings
# This saves utterance_data to a file for cross-context analysis

import numpy as np

# Assuming utterance_data is already in memory from previous cell
# utterance_data structure: {context: {idx: {layer: embedding}}}

print("Saving utterance embeddings...")

# Flatten to NPZ format
utterance_embs_dict = {}

for context in utterance_data:
    for utt_idx in utterance_data[context]:
        for layer_idx in utterance_data[context][utt_idx]:
            key = f"{context}_{utt_idx}_{layer_idx}"
            utterance_embs_dict[key] = utterance_data[context][utt_idx][layer_idx]

# Save
np.savez_compressed(BASE_PATH + 'utterance_embeddings_all_layers.npz', **utterance_embs_dict)

print(f"✓ Saved {len(utterance_embs_dict)} utterance embeddings")
print(f"  File: {BASE_PATH}utterance_embeddings_all_layers.npz")
