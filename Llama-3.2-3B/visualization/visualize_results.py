import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Load data
with open('probing_results.json', 'r') as f:
    probing_results = json.load(f)

with open('cosine_similarities.json', 'r') as f:
    cosine_similarities = json.load(f)

# Extract data
contexts = list(cosine_similarities.keys())
num_layers = len(probing_results['layer_accuracies'])
layers = list(range(num_layers))

# Convert to arrays
layer_accuracies = [probing_results['layer_accuracies'][str(i)] for i in layers]
layer_accuracies_shuffled = [probing_results['layer_accuracies_shuffled'][str(i)] for i in layers]

# ============================================================================
# VISUALIZATION 1: Overall Probing Accuracy Across Layers
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(layers, layer_accuracies, marker='o', linewidth=2, markersize=6, 
        label='Real Labels', color='#2E86AB', alpha=0.8)
ax.plot(layers, layer_accuracies_shuffled, marker='s', linewidth=2, markersize=6,
        label='Shuffled Labels (Baseline)', color='#A23B72', alpha=0.8, linestyle='--')

# Highlight best layer
best_layer = max(range(num_layers), key=lambda i: layer_accuracies[i])
ax.axvline(x=best_layer, color='red', linestyle=':', alpha=0.5, linewidth=2)
ax.text(best_layer, max(layer_accuracies) + 0.02, f'Best: Layer {best_layer}\n({layer_accuracies[best_layer]:.3f})', 
        ha='center', fontsize=10, color='red', fontweight='bold')

ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Classification Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Probing Classifier Accuracy Across Layers\n(Multi-class: 11 Emotions)', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xticks(layers)

plt.tight_layout()
plt.savefig('probing_accuracy_overall.png', dpi=300, bbox_inches='tight')
print("✓ Saved: probing_accuracy_overall.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: Per-Context Accuracy Heatmap
# ============================================================================
# Build heatmap data
context_layer_acc = probing_results['context_layer_accuracies']
heatmap_data = np.zeros((len(contexts), num_layers))

for i, context in enumerate(contexts):
    for layer in range(num_layers):
        heatmap_data[i, layer] = context_layer_acc[context][str(layer)]

fig, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(heatmap_data, annot=False, cmap='RdYlGn', cbar_kws={'label': 'Accuracy'},
            xticklabels=layers, yticklabels=contexts, vmin=0, vmax=1, ax=ax)

ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Emotion Context', fontsize=12, fontweight='bold')
ax.set_title('Per-Context Classification Accuracy Across Layers\n(How well each emotion is classified at each layer)', 
             fontsize=14, fontweight='bold', pad=20)

# Mark best layer for each context
for i, context in enumerate(contexts):
    best_layer_for_context = max(range(num_layers), 
                                  key=lambda l: context_layer_acc[context][str(l)])
    ax.add_patch(plt.Rectangle((best_layer_for_context, i), 1, 1, 
                               fill=False, edgecolor='blue', lw=3))

plt.tight_layout()
plt.savefig('probing_accuracy_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: probing_accuracy_heatmap.png")
plt.close()

# ============================================================================
# VISUALIZATION 3: Best Layers Per Context (Bar Chart)
# ============================================================================
best_layers_per_context = probing_results['best_layers_per_context']
contexts_sorted = sorted(contexts, key=lambda c: best_layers_per_context[c]['accuracy'], reverse=True)

fig, ax = plt.subplots(figsize=(12, 8))

best_layers_list = [best_layers_per_context[c]['layer'] for c in contexts_sorted]
accuracies_list = [best_layers_per_context[c]['accuracy'] for c in contexts_sorted]

# Convert layer strings to integers for plotting
best_layers_int = [int(layer) for layer in best_layers_list]

bars = ax.barh(contexts_sorted, accuracies_list, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(contexts_sorted))))

# Add layer numbers on bars
for i, (bar, layer) in enumerate(zip(bars, best_layers_int)):
    width = bar.get_width()
    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
            f'L{layer}', ha='left', va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Best Accuracy', fontsize=12, fontweight='bold')
ax.set_ylabel('Emotion Context', fontsize=12, fontweight='bold')
ax.set_title('Best Performing Layer for Each Emotion\n(Highest classification accuracy)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0, 1.05)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('best_layers_per_context.png', dpi=300, bbox_inches='tight')
print("✓ Saved: best_layers_per_context.png")
plt.close()

# ============================================================================
# VISUALIZATION 4: Cosine Similarity Distribution Across Layers
# ============================================================================
# Calculate mean cosine similarity per context per layer
mean_cosine_sim = np.zeros((len(contexts), num_layers))

for i, context in enumerate(contexts):
    for layer in range(num_layers):
        sims = cosine_similarities[context][str(layer)]
        mean_cosine_sim[i, layer] = np.mean(sims)

fig, ax = plt.subplots(figsize=(14, 6))

for i, context in enumerate(contexts):
    ax.plot(layers, mean_cosine_sim[i, :], marker='o', linewidth=1.5, 
            markersize=4, label=context, alpha=0.7)

ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Cosine Similarity', fontsize=12, fontweight='bold')
ax.set_title('Mean Prototype-Prompt Cosine Similarity Across Layers\n(Average similarity between context prototype and prompts)', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=9, loc='best', ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xticks(layers)

plt.tight_layout()
plt.savefig('cosine_similarity_by_layer.png', dpi=300, bbox_inches='tight')
print("✓ Saved: cosine_similarity_by_layer.png")
plt.close()

# ============================================================================
# VISUALIZATION 5: Cosine Similarity Heatmap
# ============================================================================
fig, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(mean_cosine_sim, annot=False, cmap='coolwarm', cbar_kws={'label': 'Mean Cosine Similarity'},
            xticklabels=layers, yticklabels=contexts, vmin=0.5, vmax=1.0, ax=ax)

ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Emotion Context', fontsize=12, fontweight='bold')
ax.set_title('Mean Cosine Similarity Heatmap\n(Prototype-Prompt similarity per context and layer)', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('cosine_similarity_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: cosine_similarity_heatmap.png")
plt.close()

# ============================================================================
# VISUALIZATION 6: Combined View - Accuracy vs Cosine Similarity
# ============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Top: Probing accuracy
ax1.plot(layers, layer_accuracies, marker='o', linewidth=2, markersize=6, 
         label='Probing Accuracy', color='#2E86AB', alpha=0.8)
ax1.set_ylabel('Classification Accuracy', fontsize=11, fontweight='bold')
ax1.set_title('Probing Accuracy vs Mean Cosine Similarity Across Layers', 
              fontsize=14, fontweight='bold', pad=20)
ax1.legend(fontsize=10, loc='lower right')
ax1.grid(True, alpha=0.3)

# Bottom: Mean cosine similarity (averaged across all contexts)
overall_mean_cosine = mean_cosine_sim.mean(axis=0)
ax2.plot(layers, overall_mean_cosine, marker='s', linewidth=2, markersize=6,
         label='Mean Cosine Similarity (All Contexts)', color='#F18F01', alpha=0.8)
ax2.set_xlabel('Layer Index', fontsize=11, fontweight='bold')
ax2.set_ylabel('Mean Cosine Similarity', fontsize=11, fontweight='bold')
ax2.legend(fontsize=10, loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(layers)

plt.tight_layout()
plt.savefig('accuracy_vs_cosine_similarity.png', dpi=300, bbox_inches='tight')
print("✓ Saved: accuracy_vs_cosine_similarity.png")
plt.close()

# ============================================================================
# VISUALIZATION 7: Cosine Similarity Violin Plot (Selected Layers)
# ============================================================================
# Select interesting layers: early, middle, late, and best
selected_layers = [0, 5, 9, 15, 20, 28, best_layer]
selected_layers = sorted(list(set(selected_layers)))  # Remove duplicates

fig, ax = plt.subplots(figsize=(14, 6))

data_for_violin = []
labels_for_violin = []

for layer in selected_layers:
    for context in contexts:
        sims = cosine_similarities[context][str(layer)]
        data_for_violin.extend(sims)
        labels_for_violin.extend([f'L{layer}'] * len(sims))

import pandas as pd
df_violin = pd.DataFrame({'Layer': labels_for_violin, 'Cosine Similarity': data_for_violin})

sns.violinplot(data=df_violin, x='Layer', y='Cosine Similarity', ax=ax, palette='Set2')
ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
ax.set_ylabel('Cosine Similarity', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Cosine Similarities Across Selected Layers\n(All contexts combined)', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('cosine_similarity_violin.png', dpi=300, bbox_inches='tight')
print("✓ Saved: cosine_similarity_violin.png")
plt.close()

# ============================================================================
# Print Summary Statistics
# ============================================================================
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print(f"\nBest Overall Layer: {best_layer} (Accuracy: {layer_accuracies[best_layer]:.4f})")
print(f"Worst Overall Layer: {np.argmin(layer_accuracies)} (Accuracy: {min(layer_accuracies):.4f})")

print("\nBest Layer per Context:")
for context in contexts:
    best_info = best_layers_per_context[context]
    print(f"  {context:12s}: Layer {best_info['layer']:2s} (Accuracy: {best_info['accuracy']:.4f})")

print("\nMean Cosine Similarity by Layer (averaged across all contexts):")
for layer in [0, 5, 10, 15, 20, 25, 28]:
    print(f"  Layer {layer:2d}: {overall_mean_cosine[layer]:.4f}")

print("\n✓ All visualizations saved!")
print("="*60)
