# Methodology Summary: Empathy-LLM Representation Analysis

## Research Question

**How does Llama-3.2-3B internally represent and process emotional content, and does its emotional understanding connect to its empathetic response generation?**

---

## Model

- **Model:** Meta Llama-3.2-3B (causal language model)
- **Architecture:** 29 transformer layers, 3072-dimensional hidden states
- **Precision:** float16 (half-precision for memory efficiency)
- **Platform:** GPU-accelerated (CUDA)

---

## Dataset

- **Source:** EmpatheticDialogues dataset (Facebook Research)
- **Emotions:** 11 emotional contexts — afraid, angry, anxious, devastated, lonely, sad, terrified, furious, grateful, hopeful, faithful
- **Prompt set (Step 1):** 100 unique emotional situation prompts per context (1100 total)
- **Utterance set (Step 3):** 100 unique first-utterance responses per context (1100 total, non-overlapping with prompt set)

---

## Pipeline Overview

| Step | What | Script |
|------|------|--------|
| **Step 1** | Data preprocessing & prompt selection | `DataPreprocessing.py` |
| **Step 2** | Embedding extraction + initial probing + L1/L2 probing | `Rep-Embeddings.py`, `L1-probing.py` |
| **Step 3** | Utterance embeddings + pairwise similarity + SCCA | `prompt-utterance-similarity.py`, `SSCA.py` |

---

## Step 1: Data Preprocessing

### What we did
1. Loaded EmpatheticDialogues train and test splits
2. Filtered for 11 target emotional contexts
3. Sampled 100 unique prompts per context (1100 total)
4. Saved as `context_prompts.json` — a dictionary mapping each emotion to its 100 prompts

### Why we did it
- Standardized sample size ensures balanced comparison across emotions
- 100 prompts per class provides sufficient statistical power for classification and similarity analyses
- JSON format enables efficient loading across all downstream scripts

### Variables
- **Input:** `combined_filtered.csv` (full EmpatheticDialogues filtered for 11 emotions)
- **Output:** `context_prompts.json` — {emotion: [100 prompts]}

---

## Step 2A: Embedding Extraction

### What we did
1. Passed each of 1100 prompts through Llama-3.2-3B with `output_hidden_states=True`
2. Extracted hidden states from all 29 layers
3. Applied **mean pooling** across token positions to get a single 3072-dim vector per layer per prompt
4. Computed **context embeddings** by averaging all 100 prompt embeddings within each emotion

### Why we did it
- Hidden states capture the model's internal representation at each processing stage
- Mean pooling aggregates variable-length sequences into fixed-size vectors
- Per-layer extraction enables tracking how representations evolve through the network
- Context (prototype) embeddings serve as centroids for similarity analysis

### Variables
- **Input:** 1100 prompts, Llama-3.2-3B model
- **Output:**
  - `prompt_embeddings.json` — shape: {context: {prompt_idx: {layer: [3072-dim vector]}}}
  - `context_embeddings` — shape: {context: {layer: [3072-dim mean vector]}}
- **Dimensions:** 1100 samples × 29 layers × 3072 features

---

## Step 2B: Initial Probing Classification

### What we did
1. For each of the 29 layers, extracted the 3072-dim embedding for all 1100 prompts
2. Split data 80/20 (stratified by emotion)
3. Trained a **Logistic Regression classifier** (L2 penalty, default) to predict emotion from embeddings
4. Trained a **shuffled baseline** (same classifier, randomized labels) to validate signal
5. Computed per-emotion accuracy using confusion matrices

### Why we did it
- **Probing classifiers** test whether specific information (emotion) is linearly decodable from a layer's representation
- If accuracy >> chance (9.1%), the layer contains emotion-discriminative information
- The shuffled baseline confirms accuracy reflects genuine signal, not overfitting to data structure
- Per-layer comparison reveals WHERE in the network emotion is best represented

### Variables
- **X (input):** Prompt embeddings per layer — shape (1100, 3072)
- **y (target):** Emotion labels — 11 classes
- **Train/test split:** 80/20, stratified, random_state=42
- **Classifier:** `LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)` — default L2 penalty

### What we got
- **Best layer overall:** Layer 5 (accuracy: 61.8%)
- **Shuffled baseline:** ~9–16% (chance level, confirming real signal)
- **Best per-emotion layers:** Grateful (Layer 15, 95%), Hopeful (Layer 12, 95%), Anxious (Layer 9, 85%)
- **Hardest emotions:** Afraid (45%), Furious (50%), Terrified (55%)
- **Finding:** Positive emotions are far more distinguishable than negative emotions; mid-layers (5–9) best encode emotion

---

## Step 2C: L1 and L2 Probing

### What we did
1. For each layer, trained two logistic regression classifiers:
   - **L1 probe** (Lasso): `penalty='l1', solver='saga', C=1.0`
   - **L2 probe** (Ridge): `penalty='l2', solver='lbfgs', C=1.0`
2. Measured **sparsity** of L1 models: fraction of zero coefficients
3. Extracted **top-20 active dimensions** per emotion at the best-performing layer

### Why we did it
- **L1 (Lasso)** forces most weights to zero → reveals which specific neurons encode emotion (feature selection)
- **L2 (Ridge)** uses all neurons → establishes performance ceiling
- Comparing L1 vs L2 accuracy tells us whether emotion is encoded in a few neurons (sparse) or many (distributed)
- The sparsity metric quantifies how concentrated emotion encoding is at each layer
- Top features identify candidate "emotion neurons" for mechanistic interpretability

### Variables
- **X (input):** Prompt embeddings per layer — shape (1100, 3072)
- **y (target):** Emotion labels — 11 classes
- **L1 regularization:** C=1.0, saga solver, max_iter=2000
- **L2 regularization:** C=1.0, lbfgs solver, max_iter=1000
- **Sparsity:** `(coef == 0).mean()` across all 11×3072 coefficients
- **Top features:** `np.argsort(np.abs(coef[i]))[-20:]` per emotion

### What we got
- **L1 best accuracy:** 58.6% at Layer 26
- **L2 best accuracy:** 60.5% at Layer 5 (matches initial probing)
- **Sparsity pattern:**
  - Layer 0: 100% zero (no signal)
  - Layer 10: 75.5% zero (only ~750 neurons active)
  - Layer 27: 58.5% zero (~1280 neurons active)
  - Layer 28: 92.5% zero (spike — emotion features discarded for output)
- **Shared neuron 243:** Appears in top-20 for 10/11 emotions → general "emotional valence" neuron
- **Finding:** Emotion transitions from distributed encoding (early layers, L2 wins) to sparse localized encoding (late layers, L1 catches up). L1 achieving near-L2 accuracy with ~40% of neurons proves emotion is concentrated, not smeared.

---

## Step 3A: Utterance Embedding Extraction

### What we did
1. Created a separate set of 1100 utterances (first dialogue responses, non-overlapping with prompts)
2. Passed each utterance through Llama-3.2-3B, extracting all 29 layer hidden states
3. Applied mean pooling to get 3072-dim vectors

### Why we did it
- Utterances represent the model's **output side** (what it would generate as an empathetic response)
- Comparing prompt vs utterance representations tells us if the model's understanding connects to its generation
- Non-overlapping sets ensure we measure genuine alignment, not memorization

### Variables
- **Input:** 1100 first-utterances from `response_prompts.csv`
- **Output:** `utterance_embeddings_all_layers.npz` — keyed as `{context}_{idx}_{layer}`

---

## Step 3B: Pairwise Similarity Matrices (Prototype Analysis)

### What we did
1. For each emotion and each layer, computed the **100×100 cosine similarity matrix** between all prompt embeddings and all utterance embeddings
2. Calculated mean, std, min, max similarity per matrix
3. Analyzed diagonal values (same-index prompt-utterance pairs)
4. Compared across layers and emotions

### Why we did it
- Measures **raw geometric alignment** between how the model represents emotional situations (prompts) vs. empathetic responses (utterances)
- The 100×100 matrix reveals whether alignment is consistent across all pairs or driven by a few outliers
- Layer-wise comparison shows at what depth prompt and utterance representations diverge
- If similarity is uniformly high, the model treats prompts and utterances as semantically equivalent

### Variables
- **X (prompts):** 100 prompt embeddings per context per layer — shape (100, 3072)
- **Y (utterances):** 100 utterance embeddings per context per layer — shape (100, 3072)
- **Similarity metric:** Cosine similarity
- **Output:** `prompt_utterance_similarity_matrices.npz` — 319 matrices (11 contexts × 29 layers)

### What we got
- **Best layer:** Layer 2 (99.87% mean similarity across all emotions)
- **Pattern:** Similarity decreases monotonically from Layer 2 (99.87%) to Layer 28 (~80%)
- **Interpretation:** Early layers haven't differentiated prompts from utterances yet — everything looks similar. Deeper layers create task-specific representations that distinguish input from output.
- **Per-emotion ranking:** Lonely/sad highest (94.4%), terrified/devastated lowest (92.8%)
- **Finding:** High similarity ≠ good emotion discrimination. The model makes all emotional text look alike in early layers.

---

## Step 3C: Sparse Canonical Correlation Analysis (SCCA)

### What we did
1. For each of 29 layers, extracted prompt matrix P (1100×3072) and utterance matrix U (1100×3072)
2. Standardized both matrices (zero mean, unit variance)
3. Reduced dimensionality with **PCA** (3072 → 200 components) — necessary because n=1100 < p=3072
4. Fit **PLS Canonical** model (10 latent components) — a practical SCCA approximation
5. Computed **canonical correlations** between projected prompt and utterance scores
6. Computed emotion-level centroids in canonical space
7. Calculated **within-class similarity** (same-emotion prompt↔utterance alignment)
8. Calculated **between-class similarity** (different-emotion prompt↔utterance alignment)
9. Computed **discrimination gap** = within − between

### Why we did it
- SCCA finds the **maximum correlation** between two high-dimensional views in a shared latent space
- Unlike cosine similarity (which measures raw geometric closeness), SCCA finds the **optimal projection** that maximizes alignment — revealing hidden shared structure
- The discrimination gap quantifies whether emotion-specific alignment exists (is "sad" prompt representation aligned specifically with "sad" utterance representation, and NOT with "angry" utterance?)
- This answers the key question: **Does the model's emotional understanding actually flow into its response generation?**

### Variables
- **X (prompts):** P — shape (1100, 3072) per layer, standardized, PCA-reduced to (1100, 200)
- **Y (utterances):** U — shape (1100, 3072) per layer, standardized, PCA-reduced to (1100, 200)
- **Labels:** 11 emotion classes (used for post-hoc analysis, NOT during fitting)
- **PLS Canonical:** n_components=10, max_iter=1000
- **PCA:** n_components=200, random_state=42
- **Canonical correlations:** Pearson correlation between P_scores[:,i] and U_scores[:,i] for each component
- **Within-class:** Mean cosine similarity of same-emotion centroids across prompt↔utterance canonical space
- **Between-class:** Mean cosine similarity of different-emotion centroids
- **Discrimination gap:** within − between

### What we got
- **Canonical correlation:** 0.79 (Layer 0) → 0.84 (Layer 10–15) — strong alignment throughout
- **Within-class similarity:** ~0.99 across all layers — near-perfect same-emotion alignment
- **Between-class similarity:** ~−0.09 (slightly negative) — different emotions are pushed apart
- **Discrimination gap:** ~1.08 (very high, stable across layers)
- **Best canonical correlation:** Layer 10–11 (0.84)
- **Finding:** The model maintains strong emotion-specific prompt↔utterance alignment throughout the network. Comprehension genuinely flows into generation with emotion identity preserved.

---

## Summary of Findings Across All Methods

| Method | Best Layers | Key Finding |
|--------|------------|-------------|
| **Initial Probing (L2)** | 5–9 | Emotion is linearly decodable at 62% accuracy (6× chance) |
| **L1 Probing** | 24–27 | Emotion concentrates into ~40% of neurons in late layers |
| **Cosine Similarity** | 2 | Prompt/utterance representations are near-identical early (99.87%) |
| **Pairwise Matrices** | 2 | Alignment degrades from 99.87% to 80% across depth |
| **SCCA** | 10–15 | Emotion-specific alignment between comprehension and generation is ~1.08 (strong) |

### Integrated Interpretation

1. **Layers 0–2:** Raw token processing. Everything looks similar. No emotion discrimination.
2. **Layers 3–9:** Emotion boundaries form. Probing peaks. Distributed encoding across many neurons.
3. **Layers 10–15:** Comprehension-generation alignment peaks. SCCA finds optimal shared structure.
4. **Layers 16–27:** Emotion concentrates into specialized neurons. L1 catches up to L2.
5. **Layer 28:** Output preparation. Emotion features are discarded (92.5% sparsity). Similarity drops to 80%.

### Core Conclusions

- Llama-3.2-3B **does encode emotion** — not trivially, but with structured, layer-specific representations
- Positive emotions (grateful, hopeful) are **much more separable** than negative emotions (afraid, terrified, furious)
- Emotion encoding is **distributed in mid-layers** but **concentrated into specialist neurons in late layers**
- The model's emotional understanding **genuinely connects to its response generation** (SCCA gap: 1.08)
- A **multi-layer approach** is optimal: Layer 2 for semantic matching, Layers 5–15 for emotion classification, Layers 10–15 for comprehension-generation alignment





Here is a comprehensive summary and report synthesizing all 15 cards, papers, PDFs, and attached notes on Empathy-LLM.

Research Report: Empathy in Large Language Models

This report synthesizes the literature, datasets, methodology proposals, and notes collected on empathy in artificial intelligence and large language models (LLMs). The collection covers foundational psychology, emotion processing pipelines, empirical evaluations of open-source models, prompt engineering, and automated evaluation frameworks.

1. Conceptual Foundations & Psychological Dimensions of Empathy

Empathy requires cognitive, emotional, behavioral, and moral capacities to understand and respond to the suffering of others [1]. Historically grounded in Martin Buber’s "I and Thou" relational distinction versus "I and It" disrespect [2], empathy exists along a perceptual and behavioral continuum where compassion serves as an active response to observed suffering [3].

The literature separates empathy into distinct psychological components:

Cognitive Empathy: Understanding another person’s emotional state [4]. It relies on perspective-taking and Theory of Mind (ToM) to generate an empathic response [5].

Affective Empathy: Actually sharing or simulating another person's emotional experience [6].

Compassionate Empathy: Moving from emotional recognition to offering active help, reassurance, and support [7].

Narrative language studies further demonstrate that highlighting sensory or emotional details in a narrative alters the language of responses produced by listeners without necessarily increasing their self-reported conscious empathy [8]. Furthermore, explicitly defining "empathic accuracy" in model prompts does not significantly alter generated response quality [9].

2. Emotion Benchmarks, Datasets, and Evaluation Metrics

Research on conversational agents, from early systems like Woebot [10] to modern LLMs, relies on established benchmarks and metric frameworks:

EMPATHETICDIALOGUES Benchmark: Introduced by Rashkin et al. (2019), this 25k-conversation dataset grounded in emotional situations benchmarks dialogue agents' abilities to recognize feelings and reply appropriately [11] [12]. Dialogue models trained on this benchmark are evaluated using automated metrics such as BLEU, retrieval precision (P@1100), and perplexity (PPL) [13].

Empathetic Rewriting: Sharma et al. (2021) demonstrated reinforcement learning agents trained on BERT classifiers across 3.3 million mental health interactions to rewrite low-empathy utterances into higher-empathy dialogues [14].

EPITOME Framework: Evaluates dialogue responses across three subcomponents on a 0-2 scale: Emotional Reactions (shared feelings), Interpretations (helping the speaker understand their situation), and Explorations (asking open-ended probing questions) [15].

Linguistic Metrics: Analyzed using LIWC-22 (Linguistic Inquiry and Word Count) for positive and negative emotion words [16] [17], alongside sentiment categorization across positive and negative lists [18].

3. Empirical LLM Capabilities, Positivity Bias, and Prompt Sensitivity

Recent studies evaluate whether LLMs possess inherent empathetic capabilities or merely perform surface-level stylistic simulation:

Inherent Empathy vs. Word Intensity: Maheswaran et al. (2025) evaluated five open-source LLMs (Falcon, Mistral, Llama, Gemma, Zephyr) and found that LLMs rely heavily on emotionally intense words (NRC Lexicon) to mimic empathy, producing responses ~5x longer than human baselines [19] [20]. LLMs also exhibit a marked positivity bias, matching positive reactions far better than negative emotions [19] [20].

Linguistic Empathy in Specialized Domains: Zhang et al. (2024) tested 10 rules of linguistic empathy (e.g., person form, collective reasoning, interim questioning, caring statements) in legal tenant assistance [21] [22]. Instructional and few-shot prompt engineering improved adherence across 7 of 10 empathy rules without degrading factual consistency [23] [24].

Underlying Representation versus Surface Prompting: A critical note attached to Zhang et al. highlights that while prompt sensitivity alters surface response style, it does not alter the model's underlying internal emotional representations [25].

Internal Emotion Representation Project (Project 4): A research proposal maps out the emotion processing pipeline: $\text{Input text} \rightarrow \text{Internal representation} \rightarrow \text{Response generation} \rightarrow \text{Observable behavior}$ [26]. Project 4 outlines linear probing across hidden layers and prototype cosine similarity analysis to test if internal representations are preserved during generation and pragmatically used to drive empathic behavior [27].

4. Automated Empathy Scoring and Leading Research Groups

Measuring empathy automatically remains a key goal in computational social science:

Interpretable LLM Scoring: Xie et al. (2024) developed an explainable scoring framework combining the Motivational Interviewing Treatment Integrity (MITI) Code with 15 explicit subfactors across Cognitive, Affective, and Compassionate empathy [28] [7]. Linear classifiers trained on 10 selected features achieved 53.65% accuracy, approaching fine-tuned GPT-4o-mini performance (54.69%) while making the evaluation transparent [29].

Key Researchers in the Field: Major research directions are led by prominent scientists including Maarten Sap, Louis-Philippe Morency, and Rada Mihalcea (internal representations), Stephen Roller, Yinhan Liu, and Timothy Bickmore (dialogue systems), Yejin Choi (social reasoning), Michal Kosinski, Noah Goodman, Tom Griffiths, and Alison Gopnik (Theory of Mind & cognitive comparisons), and Jamil Zaki and Jonathan Gratch (psychological foundations) [30].

Here is the complete bibliography list of all papers, publications, and workspace research notes referenced on the Empathy-LLM whiteboard:

Academic Publications & Preprints

Fitzpatrick, K. K., Darcy, A., & Vierhile, M. (2017).
Delivering cognitive behavior therapy to young adults using a conversational agent. JMIR Mental Health, 4(2), e19. [1]

Gueorguieva, E., Lau, T., Hadjiandreou, E., & Ong, D. (2024).
The language of an empathy-inducing narrative. In Proceedings of the Annual Meeting of the Cognitive Science Society (Vol. 46). [2]

Lee, Y. K., Suh, J., Zhan, H., Li, J. J., & Ong, D. C. (2024).
Large language models produce responses perceived to be empathic. In 2024 12th International Conference on Affective Computing and Intelligent Interaction (ACII) (pp. 63-71). IEEE. DOI: 10.1109/ACII63134.2024.00012. [3]

Maheswaran, A., Chua, C., & Desarkar, M. S. (2025).
Probing the inherent ability of large language models for generating empathetic responses. Indian Institute of Technology Hyderabad & Swinburne University of Technology. [4]

Rashkin, H., Smith, E. M., Li, M., & Boureau, Y.-L. (2019).
Towards empathetic open-domain conversation models: A new benchmark and dataset. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (pp. 5370–5381). [5] [6]

Riess, H. (2017).
The science of empathy. Journal of Patient Experience, 4(2), 74–77. DOI: 10.1177/2374373517699267. [7]

Sharma, A., Lin, I. W., Miner, A. S., Atkins, D. C., & Althoff, T. (2021).
Towards facilitating empathic conversations in online mental health support: A reinforcement learning approach. arXiv preprint arXiv:2101.07714. [8]

Xie, H. J., Zhang, J., Zhang, X., & Liu, K. (2024).
Scoring with large language models: A study on measuring empathy of responses in dialogues. Portland State University. [9]

Zhang, Y., Radishian, C., Brunswicker, S., Whitenack, D., & Linna Jr., D. W. (2024).
Empathetic language in LLMs under prompt engineering: A comparative study in the legal field. Procedia Computer Science / 6th International Conference on AI in Computational Linguistics (ACling 2024). [10]


Related Research Papers(LLM & Emotion): 
Methodology:
https://ieeexplore.ieee.org/abstract/document/10970292?casa_token=yEoEcQFMEJQAAAAA:GjeBk0Pc1YBw3hmT6w_Ud-q1mzbOdT-pn8SlM51RyzDTjPP0_RWsWOjBAu2AddEVD3LqikkonQ
Looks at whether LLMs can generate responses that people perceive as empathetic. Participants compared supportive responses written by GPT4, Turbo, Llama 2, Mistral, and humans across emotionally challenging scenarios. Humans judged the LLM responses as more empathetic than those written by humans and concluded that LLMs are capable of producing empathetic language and may be valuable tools for healthcare communication. However, this study only evaluated the perceived quality of the responses and did not look at whether the models internally represent emotions during language processing. This paper could support stage 3 of your methodology since we are investigating whether empathy comes from emotional representations in the hidden layers or whether it's a result of learned language patterns. 

https://aclanthology.org/2026.eacl-long.165.pdf
Examined how LLMs internally represent emotion by analyzing hidden layer embeddings across multiple datasets and prompting strategies. They used probing techniques and found LLMs develop consistent emotional representations that were shared across data sets with the strongest emotion encoding occurring in the later layers of the network. Although the models could recognize emotions, they also showed limitations when reasoning about complex emotional situations, meaning they did not always know how to use that emotional understanding to generate an appropriate response. This provides a strong foundation for stage 1 because it shows that emotions are represented inside LLMs hidden layers. 

https://aclanthology.org/2022.cl-1.7.pdf
A foundational paper on probing classifiers. Explains why researchers train simple classifiers on hidden embeddings and discusses strengths and limitations of probing methods. Good for justifying your probing classification methodology! 

https://aclanthology.org/D19-1410.pdf
Talks of sentence embedding and semantic similarity using cosine similarity. 

https://aclanthology.org/P19-1452.pdf

Explains that different layers learn different linguistic information, providing theoretical support for you layer by layer analysis of emotion representation in the models. 

https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2804309
Compared physician responses with chat gpt responses to patient questions and found that Chatgpt answers were often preferred for both its quality and empathy. Shows why empathy matters in healthcare LLMs. 

https://aclanthology.org/2020.emnlp-main.425.pdf
Epitome paper, talks about emotional reactions, interpretations, and explorations framework that you mention in methodology. 

https://aclanthology.org/2023.emnlp-main.653.pdf
This paper connects emotion perception with response generation, and could help support Stage 2. 

https://aclanthology.org/P19-1534.pdf
Empathy dataset

https://aclanthology.org/D18-1404.pdf
CARER emotion dataset

Papers on empathy, LLM, humans, and empathy metrics:
https://www.ovid.com/jnls/topicsinlanguagedisorders/fulltext/10.1097/tld.0000000000000040~theory-of-mind-and-empathy-as-multidimensional-constructs?casa_token=1j33TV0T5zAAAAAA%3AQBL8-c3VoctcIVQIpx9mKwK81u_QyY3MhnouiNZfW4RhtClGOHrY2QM88nbGjNvcSiV3tEREYgVWaqSsnnXEc7T8:
Explains how Theory of Mind and empathy are multidimensional processes that are supported by different brain systems. 
Theory of Mind(ToM): The ability to understand another person's mental state, such as what they are thinking, believing, intending, or feeling. They referred to it as “putting yourself in someone else's shoes” mentally. 
Empathy: the ability to understand and share another person's emotions. They distinguish this between: 
Cognitive Empathy: understanding how someone feels
Emotional (affective) empathy: actually sharing or experiencing some of that emotion yourself. 
The authors concluded that social behavior depends on a balance between these systems, as someone may understand another person's thoughts (ToM) without feeling compassion, or feel emotions without fully understanding the other person's perspective. 
My takeaway:
LLMs most likely exhibit something closer to cognitive empathy rather than emotional empathy. In the medical field, they may recognize the patient's concerns and generate the appropriate response, but it does not actually feel concern. It’s just predicting the most appropriate response based on patterns and data. 



https://www.nature.com/articles/s41562-024-01882-z
Looks at whether LLMs truly possess Theory of Mind or generate human like responses that mimic social reasoning, found that GPT4 often matched or exceeded human performance while relying on different mechanisms than humans. 
https://link.springer.com/article/10.1007/s10462-024-10921-0
This a survey of the current state of LLMs in healthcare, examining their application, performance and ethical/safety challenges that must be addressed. Many gaps mentioned here, could be a good starting point/motivator for our research paper. 

