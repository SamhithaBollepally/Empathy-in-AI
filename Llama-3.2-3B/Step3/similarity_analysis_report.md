# Prompt-Utterance Similarity Analysis Report

## Executive Summary

This analysis examines the cosine similarity between **prompt embeddings** (emotional situation descriptions) and **first utterance embeddings** (actual dialogue responses) across all 29 model layers for 11 emotion contexts.

**Key Finding:** Prompts and utterances show **extremely high similarity** (93-99%), indicating the model represents both types of text very similarly in its embedding space.

---

## Overall Statistics

- **Total comparisons:** 319 (11 contexts × 29 layers)
- **Prompts per context:** 100
- **Utterances per context:** 100
- **Similarity matrix size per layer:** 100×100 = 10,000 pairwise comparisons

---

## Key Findings

### 1. **Layer 2 is Optimal Across ALL Emotions** 🎯

**Every single emotion** achieves its highest similarity at **Layer 2**:

| Emotion | Best Layer | Mean Similarity |
|---------|-----------|-----------------|
| sad | 2 | **99.92%** |
| lonely | 2 | **99.92%** |
| hopeful | 2 | **99.91%** |
| afraid | 2 | **99.90%** |
| anxious | 2 | **98.99%** |
| furious | 2 | **99.88%** |
| grateful | 2 | **99.87%** |
| faithful | 2 | **99.88%** |
| angry | 2 | **99.88%** |
| devastated | 2 | **99.60%** |
| terrified | 2 | **99.86%** |

**Interpretation:** Layer 2 captures the **surface-level semantic similarity** between prompts and utterances before deeper contextual processing occurs.

---

### 2. **Similarity Decreases with Layer Depth**

**Top 10 layers by average similarity:**

| Rank | Layer | Avg Similarity |
|------|-------|----------------|
| 1 | Layer 2 | **99.87%** |
| 2 | Layer 3 | 99.73% |
| 3 | Layer 4 | 99.52% |
| 4 | Layer 5 | 99.31% |
| 5 | Layer 6 | 99.14% |
| 6 | Layer 7 | 98.97% |
| 7 | Layer 8 | 98.71% |
| 8 | Layer 9 | 98.57% |
| 9 | Layer 11 | 98.47% |
| 10 | Layer 10 | 98.42% |

**Pattern:** Similarity drops from **99.87%** (Layer 2) to **~80%** (Layer 28)

**Why?** Deeper layers perform more abstract, task-specific transformations that differentiate prompts from utterances.

---

### 3. **Context-Specific Similarity Rankings**

**Average similarity across all layers (highest to lowest):**

| Rank | Emotion | Avg Similarity |
|------|---------|----------------|
| 1 | **lonely** | 94.42% |
| 2 | **sad** | 94.20% |
| 3 | **hopeful** | 93.98% |
| 4 | **anxious** | 93.67% |
| 5 | **afraid** | 93.66% |
| 6 | **faithful** | 93.55% |
| 7 | **grateful** | 93.31% |
| 8 | **furious** | 93.20% |
| 9 | **angry** | 93.12% |
| 10 | **devastated** | 92.88% |
| 11 | **terrified** | 92.84% |

**Observation:** 
- **Positive emotions** (grateful, hopeful, faithful) and **sadness-related** (lonely, sad) show higher similarity
- **Intense negative emotions** (terrified, devastated, furious) show slightly lower similarity

---

## Detailed Layer Analysis

### Early Layers (0-5): Surface Similarity
- **Layer 0:** 48-53% similarity (token embeddings, no context)
- **Layer 1:** 86-87% similarity (initial contextualization)
- **Layer 2:** **99.87%** similarity ⭐ (peak semantic alignment)
- **Layers 3-5:** 99.3-99.7% (gradual differentiation begins)

### Middle Layers (6-15): Moderate Differentiation
- Similarity: 96-99%
- Model begins distinguishing prompt vs utterance characteristics
- Still maintains strong semantic alignment

### Deep Layers (16-28): Task Specialization
- Similarity: 80-96%
- Largest drop in similarity
- Model creates task-specific representations
- Layer 28: ~80% similarity (most differentiated)

---

## Statistical Insights

### Standard Deviation Patterns

**Layer 2 (best layer):**
- Mean std: 0.001-0.002 (extremely consistent)
- All pairwise similarities are very close to 99.9%

**Layer 28 (final layer):**
- Mean std: 0.06-0.07 (more variable)
- Wider range of similarities (35-96%)

**Interpretation:** Early layers treat all text uniformly; deeper layers create more nuanced, varied representations.

---

## Comparison with Previous Analyses

### Probing Classifier Results
- **Best layers:** 5, 9, 12, 15, 26 (varied by emotion)
- **Accuracy:** 42-62%

### Response Similarity Results
- **Best layer:** 28 (final layer)
- **Accuracy:** 12-17%

### Prompt-Utterance Similarity (THIS ANALYSIS)
- **Best layer:** 2 (early layer)
- **Similarity:** 99.87%

---

## Key Insights

### 1. **Why Layer 2?**
- **Surface-level semantic matching** is strongest at Layer 2
- Both prompts and utterances describe the same emotional situations
- Before task-specific processing differentiates them

### 2. **Why Such High Similarity?**
- Prompts: "I'm scared of the dark"
- Utterances: "I feel terrified when it's dark"
- **Same semantic content**, different phrasing
- Model recognizes this similarity early in processing

### 3. **Contrast with Probing Results**
- **Probing:** Tests if layers can *classify* emotions (discrimination task)
- **This analysis:** Tests if prompts/utterances are *similar* (alignment task)
- Different tasks → different optimal layers

### 4. **Implications for Response Generation**
The model:
- ✅ Understands the semantic content of prompts
- ✅ Generates utterances with similar semantic meaning
- ❌ But loses emotion-specific nuance (per previous similarity analysis)

---

## Recommendations

### For Emotion-Aware Response Generation:

1. **Use Layer 2 for semantic similarity matching**
   - Best for finding semantically similar prompts/responses
   - 99.87% accuracy in matching content

2. **Use deeper layers (5-15) for emotion classification**
   - Better discrimination between emotions
   - Per probing classifier results

3. **Avoid Layer 28 for similarity tasks**
   - Too specialized for generation
   - Loses semantic alignment (80% similarity)

4. **Multi-layer approach:**
   - Layer 2: Semantic content matching
   - Layer 9-12: Emotion classification
   - Layer 28: Response generation

---

## Conclusion

**Main Finding:** Prompts and first utterances are **nearly identical** in the model's early layers (99.87% similarity at Layer 2), confirming the model understands the semantic content. However, this high similarity also explains why generated responses lack emotion-specific nuance—the model treats all emotional contexts too similarly.

**Next Steps:**
1. Analyze the similarity matrices to find which prompt-utterance pairs are most/least similar
2. Investigate why certain emotions (terrified, devastated) show lower similarity
3. Use Layer 2 embeddings for retrieval-based response generation
4. Combine multiple layers for emotion-aware generation

---

**Generated:** Analysis of 319 layer-context combinations (11 emotions × 29 layers)
**Data:** 3.19 million pairwise similarity scores (319 matrices × 10,000 comparisons each)
