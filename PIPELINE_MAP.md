# Experimental Pipeline Map

## Project: Empathy-LLM — Emotion Encoding & Steering in Llama-3.2-3B-Instruct

**Research Question:** Does the model encode emotion during input processing, and can that encoding be leveraged to steer empathetic response generation?

**Model:** `meta-llama/Llama-3.2-3B-Instruct` (1 embedding layer + 28 decoder layers)

**Dataset:** EmpatheticDialogues (train + test + valid), filtered to 11 emotion contexts

---

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                       EMPATHETIC DIALOGUES DATASET                     │
 │  train.csv (84k rows) + test.csv (11k) + valid.csv (12k)              │
 │  11 emotions: afraid, angry, anxious, devastated, lonely, sad,        │
 │               terrified, furious, grateful, hopeful, faithful          │
 └──────────────────────────────────┬──────────────────────────────────────┘
                                    │
                                    ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                    STEP 1 — DATA PREPARATION                           │
 │                    & EMBEDDING EXTRACTION                              │
 │  Scripts: dataset.py, L3bI-Datapreprocessing.py                        │
 ├─────────────────────────────────────────────────────────────────────────┤
 │                                                                        │
 │  dataset.py                                                            │
 │  ├─ Combine all splits, filter to 11 target emotions                  │
 │  ├─ Deduplicate: one row per unique prompt (keep first utterance)     │
 │  ├─ Sample 400/context → split into:                                  │
 │  │   ├─ Set 1 (200 prompts/context) → prompts_set1.json              │
 │  │   └─ Set 2 (200 prompt+utterance pairs) → prompts_utterances_set2 │
 │  └─ Set 1 = probing/embedding | Set 2 = generation/steering          │
 │                                                                        │
 │  L3bI-Datapreprocessing.py                                             │
 │  ├─ Load Llama-3.2-3B-Instruct                                        │
 │  ├─ Forward Set 1 prompts → extract hidden states (all 29 layers)    │
 │  │   └─ Mean-pool over tokens → prompt_embeddings.npz                │
 │  ├─ Compute context prototypes (mean of 200 prompt embs per emotion) │
 │  │   └─ context_embeddings.npz                                        │
 │  └─ Forward Set 2 utterances → extract hidden states                  │
 │      └─ Mean-pool → utterance_embeddings.npz                          │
 │                                                                        │
 │  OUTPUTS:                                                              │
 │  ├─ prompts_set1.json              (200 prompts x 11 emotions)        │
 │  ├─ prompts_utterances_set2.json   (200 prompt+utt pairs x 11)       │
 │  ├─ Step-1/prompt_embeddings.npz   (2200 x 29 layers x 3072-dim)     │
 │  ├─ Step-1/context_embeddings.npz  (11 x 29 layers x 3072-dim)       │
 │  └─ Step-1/utterance_embeddings.npz(2200 x 29 layers x 3072-dim)     │
 └──────────────────┬──────────────────────────────────────────────────────┘
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
 ┌────────────────────┐ ┌─────────────────────────────────────────────────┐
 │   STEP 2a          │ │   STEP 2b                                       │
 │   PROBING          │ │   SCCA                                          │
 │   L3bI-Probing.py  │ │   L3bI-SCCA.py                                 │
 ├────────────────────┤ ├─────────────────────────────────────────────────┤
 │                    │ │                                                  │
 │ PURPOSE:           │ │ PURPOSE:                                        │
 │ Can we decode      │ │ Are prompt (Set 1) and utterance (Set 2)       │
 │ emotion from       │ │ representations aligned?                        │
 │ hidden states?     │ │                                                  │
 │                    │ │ ANALYSES:                                        │
 │ ANALYSES:          │ │ ├─ Pairwise cosine similarity:                  │
 │ ├─ L1 (Lasso) &   │ │ │  U_mean[i] vs P_mean[j] per layer           │
 │ │  L2 (Ridge)     │ │ │  → find best layer (max within-between gap)  │
 │ │  logistic probes│ │ ├─ Within-context SCCA:                         │
 │ │  per layer      │ │ │  prompts ↔ utterances per emotion per layer  │
 │ ├─ Best layer,    │ │ │  (sparse canonical correlation)              │
 │ │  sparsity,      │ │ └─ Between-context SCCA:                        │
 │ │  top features   │ │    all prompts vs all utterances per layer      │
 │ ├─ Context cosine │ │                                                  │
 │ │  similarity     │ │ OUTPUTS:                                        │
 │ │  per layer      │ │ ├─ Step-2/scca_results.json                     │
 │ │  (separation)   │ │ ├─ Step-2/scca_pairwise_cosine.png             │
 │ └─ Save probes    │ │ ├─ Step-2/scca_within_context.png              │
 │   for reuse       │ │ └─ Step-2/scca_between_context.png             │
 │                    │ │                                                  │
 │ OUTPUTS:           │ └─────────────────────────────────────────────────┘
 │ ├─ emotion_probes │
 │ │  .pt (L1+L2,    │
 │ │  29 layers,     │
 │ │  scalers)       │
 │ ├─ probing_       │
 │ │  results.json   │
 │ ├─ l1_vs_l2_     │
 │ │  probing.png    │
 │ ├─ context_cosine │
 │ │  _heatmap.png   │
 │ └─ context_sep    │
 │   _per_layer.png  │
 └────────┬───────────┘
          │
          │  probes trained on PROMPT embeddings
          │  reused downstream to classify RESPONSE embeddings
          │
          ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                    STEP 3 — RESPONSE GENERATION & ANALYSIS             │
 │  Scripts: L3bI-Zeroshot.py, L3bI-CoT.py, L3bI-ResponseEmbeddings.py, │
 │           Analysis.py, probe_utils.py                                  │
 ├─────────────────────────────────────────────────────────────────────────┤
 │                                                                        │
 │  3a. ZERO-SHOT GENERATION (L3bI-Zeroshot.py)                          │
 │  ├─ System: "You are a medical assistant..."                          │
 │  ├─ Input: Set 2 utterances (200/emotion)                             │
 │  ├─ Stage 1: Generate empathetic response + self-intensity (1-5)     │
 │  └─ Stage 2: Judge response expressiveness (1-5, greedy)             │
 │      → Outputs/zeroshot_responses.json                                │
 │                                                                        │
 │  3b. CHAIN-OF-THOUGHT GENERATION (L3bI-CoT.py)                       │
 │  ├─ System: "Before responding, reason through..."                   │
 │  │   (emotion identification → empathy → tone → intensity)           │
 │  ├─ Stage 1: Generate reasoning + response + self-intensity          │
 │  ├─ Stage 2: Judge response expressiveness                            │
 │  └─ Backfill: regenerate empty/malformed responses                   │
 │      → Outputs/CoT_responses.json                                     │
 │      → Step-3/CoT_reasoning_utterances.json                           │
 │      → Step-3/CoT_responses_scores.json                               │
 │                                                                        │
 │  3c. RESPONSE EMBEDDING (L3bI-ResponseEmbeddings.py)                  │
 │  ├─ Forward generated responses through the SAME model               │
 │  ├─ Masked mean-pool per layer (ignores padding)                     │
 │  └─ Save per-layer embeddings for both response sets                 │
 │      → Outputs/zeroshot_response_embeddings.npz                       │
 │      → Outputs/CoT_response_embeddings.npz                           │
 │                                                                        │
 │  3d. PROBE TRANSFER ANALYSIS (Analysis.py + probe_utils.py)           │
 │  ├─ Load Step-2 probes (L1 + L2 at their best layers)                │
 │  ├─ Predict emotion from RESPONSE embeddings                         │
 │  ├─ Target accuracy = does probe recover the expected emotion?       │
 │  └─ KEY FINDING: high prompt acc but low response acc =              │
 │     "the model encodes emotion on input but its generation           │
 │      doesn't access that encoding"                                    │
 │      → Step-3/zeroshot_emotion_predictions_{l1,l2}.json              │
 │      → Step-3/CoT_emotion_predictions_{l1,l2}.json                   │
 │                                                                        │
 │  ┌─────────────────────────────────────────────────────────┐           │
 │  │ KEY RESULT: Probe transfer gap                         │           │
 │  │  Prompts (held-out):  ~high accuracy (L2 best layer)   │           │
 │  │  Zeroshot responses:  ~low accuracy                    │           │
 │  │  CoT responses:       ~low accuracy                    │           │
 │  │  → Emission ≠ Encoding                                 │           │
 │  └─────────────────────────────────────────────────────────┘           │
 └──────────────────────────────────┬──────────────────────────────────────┘
                                    │
                                    ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                    STEP 4 — ACTIVATION STEERING                        │
 │  Scripts: L3bI-SA.py, L3bI-SA-contrastive.py,                         │
 │           L3bI-SA-contrastive-full.py,                                 │
 │           probe-analysis.py, probe-analysis-coherence.py,              │
 │           visualize.py                                                  │
 ├─────────────────────────────────────────────────────────────────────────┤
 │                                                                        │
 │  GOAL: Force the model to USE its emotion encoding during generation  │
 │  by injecting the emotion prototype into the residual stream.          │
 │                                                                        │
 │  4a. RAW STEERING (L3bI-SA.py)                                        │
 │  ├─ Congruent steering: for emotion e, add e's prototype              │
 │  │   hidden += coeff * ||hidden|| * unit_normalize(prototype_e)      │
 │  ├─ Norm-matched mode: relative nudge scales with hidden-state norm  │
 │  ├─ Layers: [1, 5, 8, 10, 15, 20, 25, 27, 28]                       │
 │  ├─ Coefficients: [0.05, 0.1, 0.2, 0.3, 0.5]                       │
 │  ├─ 20 utterances/emotion initial sweep                               │
 │  └─ Judge scores (unsteered) for each steered response               │
 │      → Outputs/Steering-initial/steered_*.json                        │
 │      → Step-4/steering_sweep_log.json                                 │
 │                                                                        │
 │  4b. CONTRASTIVE STEERING (L3bI-SA-contrastive.py)                    │
 │  ├─ d_e = prototype_e - grand_mean (removes shared "prompt-ness")    │
 │  │   hidden += coeff * ||hidden|| * unit_normalize(d_e)              │
 │  ├─ Same layers, coefficients, 20/emotion sweep                      │
 │  └─ Removes the generic direction that made raw steering fail        │
 │      → Outputs/SA2/steered_*.json                                     │
 │      → Step-4/steering_sweep_contrastive_log.json                     │
 │                                                                        │
 │  4c. CONTRASTIVE FULL RUN (L3bI-SA-contrastive-full.py)               │
 │  ├─ Confirmatory: all 200 utterances/emotion                         │
 │  ├─ Best coefficients from 4b sweep (c=0.2, c=0.3)                   │
 │  └─ Full-scale validation of contrastive steering                     │
 │      → Outputs/SA_full/steered_*.json                                 │
 │                                                                        │
 │  4d. PROBE EVALUATION (probe-analysis.py)                              │
 │  ├─ Embed steered responses (masked mean-pool, early-stop at         │
 │  │   deepest needed layer for speed)                                  │
 │  ├─ Predict emotion with L1/L2 probes → target accuracy              │
 │  ├─ Compared against unsteered baseline (same utterances)            │
 │  └─ Runs on: Steering-initial, SA2, SA_full                          │
 │      → Step-4/probe_eval*/steered_*.json                              │
 │      → Step-4/steering_probe_eval*.json (summary)                     │
 │                                                                        │
 │  4e. COHERENCE CHECK (probe-analysis-coherence.py)                    │
 │  ├─ Same as 4d PLUS repetition/degeneration score                    │
 │  ├─ Flags: high accuracy from looping emotion words ≠ real success   │
 │  └─ Points at SA_full outputs                                         │
 │      → Step-4/probe_eval_SA_full/                                     │
 │      → Step-4/steering_probe_eval_SA_full.json                        │
 │                                                                        │
 │  4f. VISUALIZATION (visualize.py)                                      │
 │  ├─ Fig 1: Zeroshot vs CoT (expressiveness + probe accuracy)         │
 │  ├─ Fig 2: Probe transfer gap (prompt → response)                    │
 │  ├─ Fig 3: Per-emotion recall (zeroshot, L2)                         │
 │  ├─ Fig 4: Confusion matrix (zeroshot, L2)                           │
 │  ├─ Fig 5: Steering grid — L2 target accuracy (layer x coeff)       │
 │  └─ Fig 6: Steering grid — judged expressiveness (layer x coeff)    │
 │      → Step-4/figures/fig{1-6}_*.png                                  │
 └─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Summary

```
EmpatheticDialogues CSV
        │
        ▼
   ┌─────────┐      ┌───────────────────────┐
   │ dataset  │─────▶│ Set 1: 200 prompts    │──▶ Prompt Embeddings ──▶ PROBING
   │   .py    │      │ Set 2: 200 prompt+utt │──▶ Utterance Embeddings ──▶ SCCA
   └─────────┘      └───────────┬───────────┘
                                │
                    Set 2 utterances used as
                    generation input for ▼
                    ┌───────────────────────────┐
                    │ Zeroshot / CoT Generation │
                    └─────────┬─────────────────┘
                              │ responses
                              ▼
                    ┌───────────────────────────┐
                    │ Response Embeddings       │
                    └─────────┬─────────────────┘
                              │
                    ┌─────────┴─────────────────┐
                    │ Probe Transfer Analysis   │  prompt probes → response embs
                    │ (low accuracy = gap)      │
                    └─────────┬─────────────────┘
                              │
                    ┌─────────┴─────────────────┐
                    │ Activation Steering        │  inject emotion prototype
                    │ Raw → Contrastive → Full  │  during generation
                    └─────────┬─────────────────┘
                              │
                    ┌─────────┴─────────────────┐
                    │ Probe Eval + Coherence     │  did steering improve
                    │ + Visualization            │  emotion recovery?
                    └───────────────────────────┘
```

---

## File Map

| Step | Script | Purpose | Key Inputs | Key Outputs |
|------|--------|---------|------------|-------------|
| 1 | `dataset.py` | Split dataset into Set 1 (prompts) & Set 2 (prompt+utterance) | `empatheticdialogues/*.csv` | `prompts_set1.json`, `prompts_utterances_set2.json` |
| 1 | `L3bI-Datapreprocessing.py` | Extract per-layer embeddings from model | Set 1 & Set 2 JSONs | `prompt_embeddings.npz`, `context_embeddings.npz`, `utterance_embeddings.npz` |
| 2 | `L3bI-Probing.py` | Train L1/L2 emotion probes per layer | `prompt_embeddings.npz` | `emotion_probes.pt`, `probing_results.json`, plots |
| 2 | `L3bI-SCCA.py` | Sparse CCA between prompts & utterances | `prompt_embeddings.npz`, `utterance_embeddings.npz` | `scca_results.json`, plots |
| 3 | `L3bI-Zeroshot.py` | Zero-shot response generation + judging | `prompts_utterances_set2.json` | `zeroshot_responses.json` |
| 3 | `L3bI-CoT.py` | Chain-of-Thought response generation + judging | `prompts_utterances_set2.json` | `CoT_responses.json` |
| 3 | `L3bI-ResponseEmbeddings.py` | Embed generated responses | `*_responses.json` | `*_response_embeddings.npz` |
| 3 | `Analysis.py` | Predict emotion from response embeddings using prompt probes | `*_response_embeddings.npz`, `emotion_probes.pt` | `*_emotion_predictions_{l1,l2}.json` |
| 4 | `L3bI-SA.py` | Raw activation steering (sweep) | `context_embeddings.npz`, Set 2 | `Steering-initial/steered_*.json` |
| 4 | `L3bI-SA-contrastive.py` | Contrastive steering (sweep, 20/emotion) | `context_embeddings.npz`, Set 2 | `SA2/steered_*.json` |
| 4 | `L3bI-SA-contrastive-full.py` | Contrastive steering (full, 200/emotion) | `context_embeddings.npz`, Set 2 | `SA_full/steered_*.json` |
| 4 | `probe-analysis.py` | Probe evaluation of steered responses | `steered_*.json`, `emotion_probes.pt` | `probe_eval*/`, `steering_probe_eval*.json` |
| 4 | `probe-analysis-coherence.py` | Probe eval + repetition/degeneration check | `SA_full/steered_*.json` | `probe_eval_SA_full/` |
| 4 | `visualize.py` | All presentation figures | All JSONs from Steps 2-4 | `Step-4/figures/fig{1-6}_*.png` |

---

## Steering Variants

| Variant | Vector | Formula | Scope |
|---------|--------|---------|-------|
| **Raw** | Unit-normalized prototype | `hidden += c * \|\|hidden\|\| * (proto_e / \|\|proto_e\|\|)` | 20/emotion sweep |
| **Contrastive** | Prototype minus grand mean | `hidden += c * \|\|hidden\|\| * ((proto_e - mean) / \|\|proto_e - mean\|\|)` | 20/emotion sweep |
| **Contrastive Full** | Same as contrastive | Same formula | 200/emotion (c=0.2, 0.3) |

**Layers tested:** 1, 5, 8, 10, 15, 20, 25, 27, 28
**Coefficients tested:** 0.05, 0.1, 0.2, 0.3, 0.5

---

## Shared Utilities

- **`probe_utils.py`** (Step-3): `load_probe()`, `predict_emotions()`, `load_response_embeddings_npz()` -- reused by Analysis.py and probe-analysis.py
