import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import csv

print("Loading data...")

# Load embeddings (we'll need to load from NPZ now)
prompt_embeddings_npz = np.load('/Volumes/Reading/Projects/Empathy-LLM/Step3/prompt_embeddings.npz')

# Load context prompts to get structure
with open('/Volumes/Reading/Projects/Empathy-LLM/Step3/context_prompts.json', 'r') as f:
    context_prompts = json.load(f)

contexts = list(context_prompts.keys())
print(f"Contexts: {contexts}")

# Reconstruct prompt_embeddings dictionary from NPZ
print("\nReconstructing embeddings from NPZ...")
prompt_embeddings = {}
for context in contexts:
    prompt_embeddings[context] = {}
    for prompt_idx in range(len(context_prompts[context])):
        prompt_embeddings[context][prompt_idx] = {}

# Parse NPZ keys to rebuild structure
for key in prompt_embeddings_npz.files:
    # Key format: "context_promptidx_layer"
    parts = key.rsplit('_', 2)
    if len(parts) == 3:
        context_part = parts[0]
        prompt_idx = int(parts[1])
        layer_idx = int(parts[2])
        
        # Find matching context (handle multi-word contexts)
        for context in contexts:
            if context_part == context:
                if context in prompt_embeddings and prompt_idx in prompt_embeddings[context]:
                    prompt_embeddings[context][prompt_idx][layer_idx] = prompt_embeddings_npz[key]
                break

# Determine number of layers
num_layers = max([max(prompt_embeddings[c][p].keys()) for c in contexts for p in prompt_embeddings[c]]) + 1
print(f"Number of layers: {num_layers}")

# ============================================================================
# Analyze Confusion for Selected Layers
# ============================================================================

# Layers to analyze: best overall, and best for low-accuracy contexts
layers_to_analyze = {
    5: "Best Overall",
    6: "Best for 'afraid'",
    8: "Best for 'devastated'",
    9: "Best for 'anxious'",
    15: "Best for 'grateful'"
}

print("\n" + "="*70)
print("CONFUSION MATRIX ANALYSIS")
print("="*70)

for layer_idx, layer_desc in layers_to_analyze.items():
    print(f"\n{'='*70}")
    print(f"LAYER {layer_idx}: {layer_desc}")
    print(f"{'='*70}")
    
    # Prepare data for this layer
    X = []
    y = []
    
    for context in contexts:
        for prompt_idx in range(len(context_prompts[context])):
            if layer_idx in prompt_embeddings[context][prompt_idx]:
                X.append(prompt_embeddings[context][prompt_idx][layer_idx])
                y.append(context)
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train classifier
    clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=contexts)
    
    # Normalize by true labels (rows) to get percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Save confusion matrix as CSV (raw counts)
    cm_df = pd.DataFrame(cm, index=contexts, columns=contexts)
    cm_df.index.name = 'True_Label'
    cm_df.to_csv(f'confusion_matrix_counts_layer_{layer_idx}.csv')
    print(f"✓ Saved: confusion_matrix_counts_layer_{layer_idx}.csv")
    
    # Save normalized confusion matrix as CSV
    cm_norm_df = pd.DataFrame(cm_normalized, index=contexts, columns=contexts)
    cm_norm_df.index.name = 'True_Label'
    cm_norm_df.to_csv(f'confusion_matrix_normalized_layer_{layer_idx}.csv')
    print(f"✓ Saved: confusion_matrix_normalized_layer_{layer_idx}.csv")
    
    # Save classification report as CSV
    report_dict = classification_report(y_test, y_pred, target_names=contexts, output_dict=True, digits=3)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(f'classification_report_layer_{layer_idx}.csv')
    print(f"✓ Saved: classification_report_layer_{layer_idx}.csv")
    
    # Print detailed analysis
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=contexts, digits=3))
    
    # Find most confused pairs
    print(f"\nMost Confused Emotion Pairs (excluding diagonal):")
    confusion_pairs = []
    for i, true_context in enumerate(contexts):
        for j, pred_context in enumerate(contexts):
            if i != j and cm_normalized[i, j] > 0.1:  # More than 10% confusion
                confusion_pairs.append({
                    'true': true_context,
                    'predicted': pred_context,
                    'rate': cm_normalized[i, j],
                    'count': cm[i, j]
                })
    
    confusion_pairs.sort(key=lambda x: x['rate'], reverse=True)
    
    # Save confusion pairs as CSV
    if confusion_pairs:
        confusion_df = pd.DataFrame(confusion_pairs)
        confusion_df.to_csv(f'confusion_pairs_layer_{layer_idx}.csv', index=False)
        print(f"✓ Saved: confusion_pairs_layer_{layer_idx}.csv")
        
        print(f"\n{'True Label':<15} {'Predicted As':<15} {'Confusion Rate':<15} {'Count'}")
        print("-" * 60)
        for pair in confusion_pairs[:10]:  # Top 10 confusions
            print(f"{pair['true']:<15} {pair['predicted']:<15} {pair['rate']:<15.1%} {pair['count']}")
    else:
        print("No significant confusions (>10%) found!")
    
    # Per-context accuracy
    per_context_acc = cm.diagonal() / cm.sum(axis=1)
    per_context_df = pd.DataFrame({
        'context': contexts,
        'accuracy': per_context_acc,
        'correct': cm.diagonal(),
        'total': cm.sum(axis=1)
    })
    per_context_df.to_csv(f'per_context_accuracy_layer_{layer_idx}.csv', index=False)
    print(f"✓ Saved: per_context_accuracy_layer_{layer_idx}.csv")
    
    print(f"\nPer-Context Accuracy:")
    for i, context in enumerate(contexts):
        print(f"  {context:<15}: {per_context_acc[i]:.1%} ({int(cm.diagonal()[i])}/{int(cm.sum(axis=1)[i])} correct)")

# ============================================================================
# Cross-Layer Confusion Analysis for Low-Accuracy Emotions
# ============================================================================

print("\n" + "="*70)
print("CROSS-LAYER ANALYSIS: Low-Accuracy Emotions")
print("="*70)

low_acc_emotions = ['afraid', 'furious', 'angry', 'terrified']

for emotion in low_acc_emotions:
    if emotion not in contexts:
        continue
    
    print(f"\n{'='*70}")
    print(f"Emotion: {emotion.upper()}")
    print(f"{'='*70}")
    
    # Track what this emotion is confused with across layers
    confusion_tracking = {other: [] for other in contexts if other != emotion}
    
    for layer_idx in range(num_layers):
        # Prepare data
        X, y = [], []
        for context in contexts:
            for prompt_idx in range(len(context_prompts[context])):
                if layer_idx in prompt_embeddings[context][prompt_idx]:
                    X.append(prompt_embeddings[context][prompt_idx][layer_idx])
                    y.append(context)
        
        X, y = np.array(X), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Get confusion for this emotion
        emotion_mask = (y_test == emotion)
        if emotion_mask.sum() > 0:
            predictions_for_emotion = y_pred[emotion_mask]
            for other in contexts:
                if other != emotion:
                    confusion_rate = (predictions_for_emotion == other).sum() / len(predictions_for_emotion)
                    confusion_tracking[other].append(confusion_rate)
    
    # Save confusion trends as CSV
    confusion_trend_df = pd.DataFrame(confusion_tracking)
    confusion_trend_df.insert(0, 'layer', range(num_layers))
    confusion_trend_df.to_csv(f'confusion_trend_{emotion}.csv', index=False)
    print(f"✓ Saved: confusion_trend_{emotion}.csv")
    
    # Save average confusion summary as CSV
    avg_confusion = {other: np.mean(rates) for other, rates in confusion_tracking.items()}
    avg_confusion_df = pd.DataFrame([
        {'confused_with': other, 'avg_confusion_rate': rate}
        for other, rate in sorted(avg_confusion.items(), key=lambda x: x[1], reverse=True)
    ])
    avg_confusion_df.to_csv(f'avg_confusion_{emotion}.csv', index=False)
    print(f"✓ Saved: avg_confusion_{emotion}.csv")
    
    # Print summary
    print(f"\nMost frequently confused with (averaged across all layers):")
    sorted_confusion = sorted(avg_confusion.items(), key=lambda x: x[1], reverse=True)
    for other, avg_rate in sorted_confusion[:5]:
        if avg_rate > 0.01:
            print(f"  {other:<15}: {avg_rate:.1%}")

print("\n" + "="*70)
print("✓ Confusion analysis complete!")
print("="*70)
