# Cross-Context Similarity Analysis Instructions

## What We Need to Do

We need to calculate **cross-context similarity** to determine if the model can actually discriminate between different emotions.

### Current Status:
✅ **Within-context similarity** calculated (11 matrices)
- afraid prompts × afraid utterances = 99.9%
- sad prompts × sad utterances = 99.9%
- etc.

❌ **Cross-context similarity** NOT calculated (110 matrices needed)
- afraid prompts × sad utterances = ???
- sad prompts × grateful utterances = ???
- etc.

---

## Steps to Complete Analysis

### **Step 1: Run in Colab Notebook** ✅ DONE

The notebook `prompt-utterance-similarity.ipynb` has been updated with a new cell (Cell 14).

**Run Cell 14** to save utterance embeddings:
```python
# Step 6: Save utterance embeddings for cross-context analysis
```

This will create: `utterance_embeddings_all_layers.npz`

---

### **Step 2: Run Cross-Context Analysis**

After Cell 14 completes, run the cross-context analysis script:

**In Colab:**
```python
!python Step3/calculate-cross-context-similarity.py
```

**Or locally (if you have the files):**
```bash
cd /Volumes/Reading/Projects/Empathy-LLM
python3 Step3/calculate-cross-context-similarity.py
```

---

## What the Analysis Will Calculate

### **11×11 Cross-Context Matrix** (for each layer)

|  | afraid_utt | angry_utt | sad_utt | grateful_utt | ... |
|---|---|---|---|---|---|
| **afraid_prompt** | 99.9% ✓ | ??? | ??? | ??? | ... |
| **angry_prompt** | ??? | 99.9% ✓ | ??? | ??? | ... |
| **sad_prompt** | ??? | ??? | 99.9% ✓ | ??? | ... |
| **grateful_prompt** | ??? | ??? | ??? | 99.9% ✓ | ... |

- **Diagonal** = within-context (same emotion)
- **Off-diagonal** = cross-context (different emotions)

---

## Expected Outputs

### **1. Files Created:**
- `cross_context_similarity_matrices.npz` - All 11×11 matrices
- `cross_context_similarity_stats.csv` - Statistics table
- `cross_context_similarity_matrices.png` - Heatmap visualization
- `discrimination_gap_analysis.png` - Gap analysis plot

### **2. Key Metrics:**

**Discrimination Gap** = Within-context similarity - Cross-context similarity

**Interpretation:**
- **Gap < 0.01** → ❌ POOR - Model cannot distinguish emotions
- **Gap 0.01-0.05** → ⚡ WEAK - Some emotion separation
- **Gap 0.05-0.10** → ✓ MODERATE discrimination
- **Gap > 0.10** → ✓✓ STRONG discrimination

---

## Expected Results (Hypothesis)

Based on the 99.9% within-context similarity at Layer 2, we expect:

### **Layer 2:**
- Within-context: 99.9%
- Cross-context: **~99.8%** (probably!)
- **Gap: ~0.001** → Model CANNOT discriminate

### **Layer 9:**
- Within-context: 98.6%
- Cross-context: **~98.0%** (maybe?)
- **Gap: ~0.006** → WEAK discrimination

### **Layer 28:**
- Within-context: 80.5%
- Cross-context: **~75%** (hopefully?)
- **Gap: ~0.05** → MODERATE discrimination

---

## Why This Matters

### **If cross-context similarity is also very high:**

This would explain:
1. ❌ Why response generation lacks emotion specificity
2. ❌ Why all responses are generic
3. ❌ Model treats all emotions as nearly identical
4. ❌ Cannot distinguish "sad" from "happy" contexts

### **If cross-context similarity is lower:**

This would show:
1. ✓ Model CAN distinguish emotions
2. ✓ Problem is in response generation, not understanding
3. ✓ Need to use deeper layers for emotion-specific generation

---

## Next Steps After Analysis

1. **Review the discrimination gap** across all layers
2. **Identify which layers** (if any) can discriminate emotions
3. **Compare with probing results** to validate findings
4. **Update response generation strategy** based on findings

---

## Files Reference

### **Scripts:**
- `prompt-utterance-similarity.ipynb` - Main analysis notebook
- `calculate-cross-context-similarity.py` - Cross-context calculator
- `visualize-pairwise-matrices.py` - Visualization script

### **Data Files:**
- `prompt_embeddings.json` - Prompt embeddings (Step 1)
- `utterance_embeddings_all_layers.npz` - Utterance embeddings (NEW)
- `prompt_utterance_similarity_matrices.npz` - Within-context matrices
- `cross_context_similarity_matrices.npz` - Cross-context matrices (TO BE CREATED)

---

## Timeline

- **Cell 14 (Save utterances):** ~30 seconds
- **Cross-context calculation:** ~10-15 minutes (4 layers × 121 matrices)
- **Visualization:** ~2 minutes

**Total:** ~15-20 minutes

---

**Ready to proceed!** Run Cell 14 in the notebook, then run the cross-context analysis script.
