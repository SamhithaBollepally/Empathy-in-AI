from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

doc = Document()

# -- Page setup --
for section in doc.sections:
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(2.54)
    section.right_margin = Cm(2.54)

# -- Style configuration --
style = doc.styles['Normal']
font = style.font
font.name = 'Calibri'
font.size = Pt(11)
style.paragraph_format.space_after = Pt(6)
style.paragraph_format.line_spacing = 1.15

for level in range(1, 4):
    heading_style = doc.styles[f'Heading {level}']
    heading_style.font.color.rgb = RGBColor(0, 51, 102)
    heading_style.font.name = 'Calibri'

doc.styles['Heading 1'].font.size = Pt(18)
doc.styles['Heading 2'].font.size = Pt(14)
doc.styles['Heading 3'].font.size = Pt(12)


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Header
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(10)
    # Rows
    for r_idx, row_data in enumerate(rows):
        for c_idx, val in enumerate(row_data):
            cell = table.rows[r_idx + 1].cells[c_idx]
            cell.text = str(val)
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(10)
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Cm(w)
    doc.add_paragraph()
    return table


def add_bullet(doc, text, bold_prefix=None):
    p = doc.add_paragraph(style='List Bullet')
    if bold_prefix:
        run = p.add_run(bold_prefix)
        run.bold = True
        p.add_run(text)
    else:
        p.add_run(text)
    return p


# ============================================================
# TITLE PAGE
# ============================================================
for _ in range(6):
    doc.add_paragraph()

title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run('Empathy in Large Language Models:\nInternal Representation Analysis')
run.font.size = Pt(26)
run.bold = True
run.font.color.rgb = RGBColor(0, 51, 102)

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = subtitle.add_run('A Comprehensive Methodology and Results Report')
run.font.size = Pt(14)
run.font.color.rgb = RGBColor(80, 80, 80)

doc.add_paragraph()
model_info = doc.add_paragraph()
model_info.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = model_info.add_run('Model: Meta Llama-3.2-3B\nDataset: EmpatheticDialogues (Facebook Research)')
run.font.size = Pt(12)
run.font.color.rgb = RGBColor(100, 100, 100)

doc.add_page_break()

# ============================================================
# TABLE OF CONTENTS placeholder
# ============================================================
doc.add_heading('Table of Contents', level=1)
toc_items = [
    '1. Introduction & Research Question',
    '2. Literature Review',
    '   2.1 Conceptual Foundations of Empathy',
    '   2.2 Emotion Benchmarks, Datasets, and Evaluation Metrics',
    '   2.3 Empirical LLM Capabilities and Prompt Sensitivity',
    '   2.4 Automated Empathy Scoring',
    '3. Model & Dataset',
    '4. Methodology',
    '   4.1 Step 1: Data Preprocessing',
    '   4.2 Step 2A: Embedding Extraction',
    '   4.3 Step 2B: Initial Probing Classification',
    '   4.4 Step 2C: L1 and L2 Probing',
    '   4.5 Step 3A: Utterance Embedding Extraction',
    '   4.6 Step 3B: Pairwise Similarity Matrices (Prototype Analysis)',
    '   4.7 Step 3C: Sparse Canonical Correlation Analysis (SCCA)',
    '5. Results',
    '   5.1 Initial Probing Results',
    '   5.2 L1/L2 Probing Results',
    '   5.3 Pairwise Similarity Results',
    '   5.4 SCCA Results',
    '   5.5 Cross-Method Comparison',
    '6. Integrated Interpretation',
    '7. Core Conclusions',
    '8. Related Work & References',
    '9. Bibliography',
]
for item in toc_items:
    p = doc.add_paragraph(item)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.space_before = Pt(0)

doc.add_page_break()

# ============================================================
# 1. INTRODUCTION
# ============================================================
doc.add_heading('1. Introduction & Research Question', level=1)

doc.add_paragraph(
    'This report presents a comprehensive analysis of how the large language model '
    'Llama-3.2-3B internally represents and processes emotional content, and whether '
    'its emotional understanding connects to its empathetic response generation. '
    'The research bridges the gap between observed empathetic behavior in LLM outputs '
    'and the internal computational mechanisms that produce such behavior.'
)

doc.add_paragraph(
    'The central research question is:'
)

rq = doc.add_paragraph()
rq.alignment = WD_ALIGN_PARAGRAPH.CENTER
rq.paragraph_format.space_before = Pt(6)
rq.paragraph_format.space_after = Pt(6)
run = rq.add_run(
    '"How does Llama-3.2-3B internally represent and process emotional content, '
    'and does its emotional understanding connect to its empathetic response generation?"'
)
run.italic = True
run.font.size = Pt(12)

doc.add_paragraph(
    'To answer this question, we employ a multi-method approach spanning probing classifiers, '
    'sparse feature analysis, pairwise cosine similarity matrices, and Sparse Canonical '
    'Correlation Analysis (SCCA) across all 29 transformer layers of the model. The emotion '
    'processing pipeline under investigation follows the path: Input text -> Internal '
    'representation -> Response generation -> Observable behavior.'
)

# ============================================================
# 2. LITERATURE REVIEW
# ============================================================
doc.add_heading('2. Literature Review', level=1)

doc.add_paragraph(
    'This section synthesizes the literature, datasets, methodology proposals, and notes '
    'collected on empathy in artificial intelligence and large language models (LLMs). '
    'The collection covers foundational psychology, emotion processing pipelines, empirical '
    'evaluations of open-source models, prompt engineering, and automated evaluation frameworks.'
)

# 2.1
doc.add_heading('2.1 Conceptual Foundations & Psychological Dimensions of Empathy', level=2)

doc.add_paragraph(
    'Empathy requires cognitive, emotional, behavioral, and moral capacities to understand '
    'and respond to the suffering of others [1]. Historically grounded in Martin Buber\'s '
    '"I and Thou" relational distinction versus "I and It" disrespect [2], empathy exists '
    'along a perceptual and behavioral continuum where compassion serves as an active '
    'response to observed suffering [3].'
)

doc.add_paragraph('The literature separates empathy into distinct psychological components:')

add_bullet(doc, ' Understanding another person\'s emotional state. It relies on '
           'perspective-taking and Theory of Mind (ToM) to generate an empathic response [4][5].',
           'Cognitive Empathy:')
add_bullet(doc, ' Actually sharing or simulating another person\'s emotional experience [6].',
           'Affective Empathy:')
add_bullet(doc, ' Moving from emotional recognition to offering active help, reassurance, '
           'and support [7].',
           'Compassionate Empathy:')

doc.add_paragraph(
    'Narrative language studies further demonstrate that highlighting sensory or emotional '
    'details in a narrative alters the language of responses produced by listeners without '
    'necessarily increasing their self-reported conscious empathy [8]. Furthermore, explicitly '
    'defining "empathic accuracy" in model prompts does not significantly alter generated '
    'response quality [9].'
)

# 2.2
doc.add_heading('2.2 Emotion Benchmarks, Datasets, and Evaluation Metrics', level=2)

doc.add_paragraph(
    'Research on conversational agents, from early systems like Woebot [10] to modern LLMs, '
    'relies on established benchmarks and metric frameworks:'
)

add_bullet(doc, ' Introduced by Rashkin et al. (2019), this 25k-conversation dataset '
           'grounded in emotional situations benchmarks dialogue agents\' abilities to recognize '
           'feelings and reply appropriately [11][12]. Dialogue models trained on this benchmark '
           'are evaluated using automated metrics such as BLEU, retrieval precision (P@1100), '
           'and perplexity (PPL) [13].',
           'EmpatheticDialogues Benchmark:')
add_bullet(doc, ' Sharma et al. (2021) demonstrated reinforcement learning agents '
           'trained on BERT classifiers across 3.3 million mental health interactions to rewrite '
           'low-empathy utterances into higher-empathy dialogues [14].',
           'Empathetic Rewriting:')
add_bullet(doc, ' Evaluates dialogue responses across three subcomponents on a 0-2 '
           'scale: Emotional Reactions (shared feelings), Interpretations (helping the speaker '
           'understand their situation), and Explorations (asking open-ended probing questions) [15].',
           'EPITOME Framework:')
add_bullet(doc, ' Analyzed using LIWC-22 (Linguistic Inquiry and Word Count) for positive '
           'and negative emotion words [16][17], alongside sentiment categorization across '
           'positive and negative lists [18].',
           'Linguistic Metrics:')

# 2.3
doc.add_heading('2.3 Empirical LLM Capabilities, Positivity Bias, and Prompt Sensitivity', level=2)

doc.add_paragraph(
    'Recent studies evaluate whether LLMs possess inherent empathetic capabilities or merely '
    'perform surface-level stylistic simulation:'
)

add_bullet(doc, ' Maheswaran et al. (2025) evaluated five open-source LLMs '
           '(Falcon, Mistral, Llama, Gemma, Zephyr) and found that LLMs rely heavily on '
           'emotionally intense words (NRC Lexicon) to mimic empathy, producing responses ~5x '
           'longer than human baselines [19][20]. LLMs also exhibit a marked positivity bias, '
           'matching positive reactions far better than negative emotions.',
           'Inherent Empathy vs. Word Intensity:')
add_bullet(doc, ' Zhang et al. (2024) tested 10 rules of linguistic empathy '
           '(e.g., person form, collective reasoning, interim questioning, caring statements) in '
           'legal tenant assistance [21][22]. Instructional and few-shot prompt engineering improved '
           'adherence across 7 of 10 empathy rules without degrading factual consistency [23][24].',
           'Linguistic Empathy in Specialized Domains:')
add_bullet(doc, ' A critical note highlights that while prompt '
           'sensitivity alters surface response style, it does not alter the model\'s underlying '
           'internal emotional representations [25].',
           'Underlying Representation vs. Surface Prompting:')

# 2.4
doc.add_heading('2.4 Automated Empathy Scoring and Leading Research Groups', level=2)

doc.add_paragraph(
    'Xie et al. (2024) developed an explainable scoring framework combining the Motivational '
    'Interviewing Treatment Integrity (MITI) Code with 15 explicit subfactors across Cognitive, '
    'Affective, and Compassionate empathy [28][7]. Linear classifiers trained on 10 selected '
    'features achieved 53.65% accuracy, approaching fine-tuned GPT-4o-mini performance (54.69%) '
    'while making the evaluation transparent [29].'
)

doc.add_paragraph(
    'Major research directions in this field are led by prominent scientists including '
    'Maarten Sap, Louis-Philippe Morency, and Rada Mihalcea (internal representations), '
    'Stephen Roller, Yinhan Liu, and Timothy Bickmore (dialogue systems), Yejin Choi '
    '(social reasoning), Michal Kosinski, Noah Goodman, Tom Griffiths, and Alison Gopnik '
    '(Theory of Mind & cognitive comparisons), and Jamil Zaki and Jonathan Gratch '
    '(psychological foundations) [30].'
)

doc.add_page_break()

# ============================================================
# 3. MODEL & DATASET
# ============================================================
doc.add_heading('3. Model & Dataset', level=1)

doc.add_heading('3.1 Model', level=2)

add_table(doc,
    ['Parameter', 'Value'],
    [
        ['Model', 'Meta Llama-3.2-3B (causal language model)'],
        ['Architecture', '29 transformer layers, 3072-dimensional hidden states'],
        ['Precision', 'float16 (half-precision for memory efficiency)'],
        ['Platform', 'GPU-accelerated (CUDA)'],
    ],
    col_widths=[5, 12]
)

doc.add_heading('3.2 Dataset', level=2)

add_table(doc,
    ['Parameter', 'Value'],
    [
        ['Source', 'EmpatheticDialogues dataset (Facebook Research)'],
        ['Emotions', '11 contexts: afraid, angry, anxious, devastated, lonely, sad, terrified, furious, grateful, hopeful, faithful'],
        ['Prompt set (Step 1)', '100 unique emotional situation prompts per context (1100 total)'],
        ['Utterance set (Step 3)', '100 unique first-utterance responses per context (1100 total, non-overlapping with prompt set)'],
    ],
    col_widths=[5, 12]
)

doc.add_heading('3.3 Pipeline Overview', level=2)

add_table(doc,
    ['Step', 'Description', 'Script(s)'],
    [
        ['Step 1', 'Data preprocessing & prompt selection', 'DataPreprocessing.py'],
        ['Step 2', 'Embedding extraction + initial probing + L1/L2 probing', 'Rep-Embeddings.py, L1-probing.py'],
        ['Step 3', 'Utterance embeddings + pairwise similarity + SCCA', 'prompt-utterance-similarity.py, SSCA.py'],
    ],
    col_widths=[2.5, 8, 6.5]
)

# ============================================================
# 4. METHODOLOGY
# ============================================================
doc.add_heading('4. Methodology', level=1)

# 4.1
doc.add_heading('4.1 Step 1: Data Preprocessing', level=2)

doc.add_heading('What we did', level=3)
doc.add_paragraph(
    '1. Loaded EmpatheticDialogues train and test splits.\n'
    '2. Filtered for 11 target emotional contexts.\n'
    '3. Sampled 100 unique prompts per context (1100 total).\n'
    '4. Saved as context_prompts.json \u2014 a dictionary mapping each emotion to its 100 prompts.'
)

doc.add_heading('Why we did it', level=3)
add_bullet(doc, 'Standardized sample size ensures balanced comparison across emotions.')
add_bullet(doc, '100 prompts per class provides sufficient statistical power for classification and similarity analyses.')
add_bullet(doc, 'JSON format enables efficient loading across all downstream scripts.')

doc.add_heading('Variables', level=3)
add_table(doc,
    ['Variable', 'Description'],
    [
        ['Input', 'combined_filtered.csv (full EmpatheticDialogues filtered for 11 emotions)'],
        ['Output', 'context_prompts.json \u2014 {emotion: [100 prompts]}'],
    ],
    col_widths=[4, 13]
)

# 4.2
doc.add_heading('4.2 Step 2A: Embedding Extraction', level=2)

doc.add_heading('What we did', level=3)
doc.add_paragraph(
    '1. Passed each of 1100 prompts through Llama-3.2-3B with output_hidden_states=True.\n'
    '2. Extracted hidden states from all 29 layers.\n'
    '3. Applied mean pooling across token positions to get a single 3072-dimensional vector per layer per prompt.\n'
    '4. Computed context embeddings by averaging all 100 prompt embeddings within each emotion.'
)

doc.add_heading('Why we did it', level=3)
add_bullet(doc, 'Hidden states capture the model\'s internal representation at each processing stage.')
add_bullet(doc, 'Mean pooling aggregates variable-length sequences into fixed-size vectors.')
add_bullet(doc, 'Per-layer extraction enables tracking how representations evolve through the network.')
add_bullet(doc, 'Context (prototype) embeddings serve as centroids for similarity analysis.')

doc.add_heading('Variables', level=3)
add_table(doc,
    ['Variable', 'Description'],
    [
        ['Input', '1100 prompts, Llama-3.2-3B model'],
        ['Output: prompt_embeddings.json', '{context: {prompt_idx: {layer: [3072-dim vector]}}}'],
        ['Output: context_embeddings', '{context: {layer: [3072-dim mean vector]}}'],
        ['Dimensions', '1100 samples x 29 layers x 3072 features'],
    ],
    col_widths=[6, 11]
)

# 4.3
doc.add_heading('4.3 Step 2B: Initial Probing Classification', level=2)

doc.add_heading('What we did', level=3)
doc.add_paragraph(
    '1. For each of the 29 layers, extracted the 3072-dim embedding for all 1100 prompts.\n'
    '2. Split data 80/20 (stratified by emotion).\n'
    '3. Trained a Logistic Regression classifier (L2 penalty, default) to predict emotion from embeddings.\n'
    '4. Trained a shuffled baseline (same classifier, randomized labels) to validate signal.\n'
    '5. Computed per-emotion accuracy using confusion matrices.'
)

doc.add_heading('Why we did it', level=3)
add_bullet(doc, 'Probing classifiers test whether specific information (emotion) is linearly decodable from a layer\'s representation.')
add_bullet(doc, 'If accuracy >> chance (9.1%), the layer contains emotion-discriminative information.')
add_bullet(doc, 'The shuffled baseline confirms accuracy reflects genuine signal, not overfitting.')
add_bullet(doc, 'Per-layer comparison reveals where in the network emotion is best represented.')

doc.add_heading('Variables', level=3)
add_table(doc,
    ['Variable', 'Description'],
    [
        ['X (input)', 'Prompt embeddings per layer \u2014 shape (1100, 3072)'],
        ['y (target)', 'Emotion labels \u2014 11 classes'],
        ['Train/test split', '80/20, stratified, random_state=42'],
        ['Classifier', 'LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1) \u2014 default L2 penalty'],
    ],
    col_widths=[4, 13]
)

# 4.4
doc.add_heading('4.4 Step 2C: L1 and L2 Probing', level=2)

doc.add_heading('What we did', level=3)
doc.add_paragraph(
    '1. For each layer, trained two logistic regression classifiers:\n'
    '   \u2022 L1 probe (Lasso): penalty=\'l1\', solver=\'saga\', C=1.0, max_iter=2000\n'
    '   \u2022 L2 probe (Ridge): penalty=\'l2\', solver=\'lbfgs\', C=1.0, max_iter=1000\n'
    '2. Measured sparsity of L1 models: fraction of zero coefficients.\n'
    '3. Extracted top-20 active dimensions per emotion at the best-performing layer.'
)

doc.add_heading('Why we did it', level=3)
add_bullet(doc, 'L1 (Lasso) forces most weights to zero, revealing which specific neurons encode emotion (feature selection).')
add_bullet(doc, 'L2 (Ridge) uses all neurons, establishing a performance ceiling.')
add_bullet(doc, 'Comparing L1 vs. L2 accuracy determines whether emotion is encoded sparsely (few neurons) or distributed (many neurons).')
add_bullet(doc, 'The sparsity metric quantifies how concentrated emotion encoding is at each layer.')
add_bullet(doc, 'Top features identify candidate "emotion neurons" for mechanistic interpretability.')

doc.add_heading('Variables', level=3)
add_table(doc,
    ['Variable', 'Description'],
    [
        ['X (input)', 'Prompt embeddings per layer \u2014 shape (1100, 3072)'],
        ['y (target)', 'Emotion labels \u2014 11 classes'],
        ['L1 regularization', 'C=1.0, saga solver, max_iter=2000'],
        ['L2 regularization', 'C=1.0, lbfgs solver, max_iter=1000'],
        ['Sparsity', '(coef == 0).mean() across all 11x3072 coefficients'],
        ['Top features', 'np.argsort(np.abs(coef[i]))[-20:] per emotion'],
    ],
    col_widths=[5, 12]
)

# 4.5
doc.add_heading('4.5 Step 3A: Utterance Embedding Extraction', level=2)

doc.add_heading('What we did', level=3)
doc.add_paragraph(
    '1. Created a separate set of 1100 utterances (first dialogue responses, non-overlapping with prompts).\n'
    '2. Passed each utterance through Llama-3.2-3B, extracting all 29 layer hidden states.\n'
    '3. Applied mean pooling to get 3072-dim vectors.'
)

doc.add_heading('Why we did it', level=3)
add_bullet(doc, 'Utterances represent the model\'s output side (what it would generate as an empathetic response).')
add_bullet(doc, 'Comparing prompt vs. utterance representations tells us if the model\'s understanding connects to its generation.')
add_bullet(doc, 'Non-overlapping sets ensure we measure genuine alignment, not memorization.')

doc.add_heading('Variables', level=3)
add_table(doc,
    ['Variable', 'Description'],
    [
        ['Input', '1100 first-utterances from response_prompts.csv'],
        ['Output', 'utterance_embeddings_all_layers.npz \u2014 keyed as {context}_{idx}_{layer}'],
    ],
    col_widths=[4, 13]
)

# 4.6
doc.add_heading('4.6 Step 3B: Pairwise Similarity Matrices (Prototype Analysis)', level=2)

doc.add_heading('What we did', level=3)
doc.add_paragraph(
    '1. For each emotion and each layer, computed the 100x100 cosine similarity matrix between all prompt embeddings and all utterance embeddings.\n'
    '2. Calculated mean, std, min, max similarity per matrix.\n'
    '3. Analyzed diagonal values (same-index prompt-utterance pairs).\n'
    '4. Compared across layers and emotions.'
)

doc.add_heading('Why we did it', level=3)
add_bullet(doc, 'Measures raw geometric alignment between how the model represents emotional situations (prompts) vs. empathetic responses (utterances).')
add_bullet(doc, 'The 100x100 matrix reveals whether alignment is consistent across all pairs or driven by a few outliers.')
add_bullet(doc, 'Layer-wise comparison shows at what depth prompt and utterance representations diverge.')
add_bullet(doc, 'If similarity is uniformly high, the model treats prompts and utterances as semantically equivalent.')

doc.add_heading('Variables', level=3)
add_table(doc,
    ['Variable', 'Description'],
    [
        ['X (prompts)', '100 prompt embeddings per context per layer \u2014 shape (100, 3072)'],
        ['Y (utterances)', '100 utterance embeddings per context per layer \u2014 shape (100, 3072)'],
        ['Similarity metric', 'Cosine similarity'],
        ['Output', 'prompt_utterance_similarity_matrices.npz \u2014 319 matrices (11 contexts x 29 layers)'],
    ],
    col_widths=[5, 12]
)

# 4.7
doc.add_heading('4.7 Step 3C: Sparse Canonical Correlation Analysis (SCCA)', level=2)

doc.add_heading('What we did', level=3)
doc.add_paragraph(
    '1. For each of 29 layers, extracted prompt matrix P (1100x3072) and utterance matrix U (1100x3072).\n'
    '2. Standardized both matrices (zero mean, unit variance).\n'
    '3. Reduced dimensionality with PCA (3072 -> 200 components) \u2014 necessary because n=1100 < p=3072.\n'
    '4. Fit PLS Canonical model (10 latent components) \u2014 a practical SCCA approximation.\n'
    '5. Computed canonical correlations between projected prompt and utterance scores.\n'
    '6. Computed emotion-level centroids in canonical space.\n'
    '7. Calculated within-class similarity (same-emotion prompt-utterance alignment).\n'
    '8. Calculated between-class similarity (different-emotion prompt-utterance alignment).\n'
    '9. Computed discrimination gap = within - between.'
)

doc.add_heading('Why we did it', level=3)
add_bullet(doc, 'SCCA finds the maximum correlation between two high-dimensional views in a shared latent space.')
add_bullet(doc, 'Unlike cosine similarity (which measures raw geometric closeness), SCCA finds the optimal projection that maximizes alignment, revealing hidden shared structure.')
add_bullet(doc, 'The discrimination gap quantifies whether emotion-specific alignment exists between comprehension and generation representations.')
add_bullet(doc, 'This answers the key question: Does the model\'s emotional understanding actually flow into its response generation?')

doc.add_heading('Variables', level=3)
add_table(doc,
    ['Variable', 'Description'],
    [
        ['X (prompts)', 'P \u2014 shape (1100, 3072) per layer, standardized, PCA-reduced to (1100, 200)'],
        ['Y (utterances)', 'U \u2014 shape (1100, 3072) per layer, standardized, PCA-reduced to (1100, 200)'],
        ['Labels', '11 emotion classes (used for post-hoc analysis, NOT during fitting)'],
        ['PLS Canonical', 'n_components=10, max_iter=1000'],
        ['PCA', 'n_components=200, random_state=42'],
        ['Canonical correlations', 'Pearson correlation between P_scores[:,i] and U_scores[:,i]'],
        ['Within-class', 'Mean cosine similarity of same-emotion centroids'],
        ['Between-class', 'Mean cosine similarity of different-emotion centroids'],
        ['Discrimination gap', 'within - between'],
    ],
    col_widths=[5, 12]
)

doc.add_page_break()

# ============================================================
# 5. RESULTS
# ============================================================
doc.add_heading('5. Results', level=1)

# 5.1
doc.add_heading('5.1 Initial Probing Classification Results', level=2)

doc.add_paragraph(
    'The initial L2 probing classifier achieved a peak accuracy of 61.8% at Layer 5, '
    'approximately 6x above chance level (9.1%). The shuffled baseline remained at 9-16% '
    'across all layers, confirming the detected signal is genuine.'
)

add_table(doc,
    ['Metric', 'Value'],
    [
        ['Best layer overall', 'Layer 5 (accuracy: 61.8%)'],
        ['Shuffled baseline', '~9-16% (chance level)'],
        ['Best: Grateful', 'Layer 15, 95%'],
        ['Best: Hopeful', 'Layer 12, 95%'],
        ['Best: Anxious', 'Layer 9, 85%'],
        ['Hardest: Afraid', '45%'],
        ['Hardest: Furious', '50%'],
        ['Hardest: Terrified', '55%'],
    ],
    col_widths=[5, 12]
)

p = doc.add_paragraph()
run = p.add_run('Finding: ')
run.bold = True
p.add_run('Positive emotions are far more distinguishable than negative emotions. '
          'Mid-layers (5-9) best encode emotion for classification purposes.')

# 5.2
doc.add_heading('5.2 L1 and L2 Probing Results', level=2)

add_table(doc,
    ['Metric', 'L1 (Lasso)', 'L2 (Ridge)'],
    [
        ['Best accuracy', '58.6% at Layer 26', '60.5% at Layer 5'],
        ['Peak layers', 'Late (24-27)', 'Early-mid (5-9)'],
        ['Features used', 'Sparse (~40% of neurons)', 'All 3072 neurons'],
    ],
    col_widths=[4, 6.5, 6.5]
)

doc.add_heading('Sparsity Pattern', level=3)
add_table(doc,
    ['Layer', 'Zero Coefficients', 'Active Neurons (approx.)', 'Interpretation'],
    [
        ['0', '100%', '0', 'No signal'],
        ['10', '75.5%', '~750', 'Sparse encoding'],
        ['27', '58.5%', '~1280', 'Most concentrated'],
        ['28', '92.5%', '~230', 'Emotion discarded for output'],
    ],
    col_widths=[2, 4, 4.5, 6.5]
)

p = doc.add_paragraph()
run = p.add_run('Key finding: ')
run.bold = True
p.add_run('Dimension 243 appears in the top-20 features for 10/11 emotions, suggesting '
          'it functions as a general "emotional valence" neuron. Emotion transitions from '
          'distributed encoding (early layers, L2 wins) to sparse localized encoding '
          '(late layers, L1 catches up).')

# 5.3
doc.add_heading('5.3 Pairwise Similarity Results', level=2)

doc.add_paragraph(
    'Cosine similarity between prompt and utterance embeddings peaks at Layer 2 (99.87%) '
    'and decreases monotonically to ~80% at Layer 28.'
)

add_table(doc,
    ['Layer Range', 'Mean Similarity', 'Interpretation'],
    [
        ['Layer 0', '48-53%', 'Token embeddings, no context'],
        ['Layer 1', '86-87%', 'Initial contextualization'],
        ['Layer 2', '99.87%', 'Peak semantic alignment'],
        ['Layers 3-5', '99.3-99.7%', 'Gradual differentiation begins'],
        ['Layers 6-15', '96-99%', 'Moderate differentiation'],
        ['Layers 16-28', '80-96%', 'Task specialization'],
    ],
    col_widths=[4, 4, 9]
)

doc.add_heading('Per-Emotion Similarity Rankings (averaged across layers)', level=3)
add_table(doc,
    ['Rank', 'Emotion', 'Avg Similarity'],
    [
        ['1', 'lonely', '94.42%'],
        ['2', 'sad', '94.20%'],
        ['3', 'hopeful', '93.98%'],
        ['4', 'anxious', '93.67%'],
        ['5', 'afraid', '93.66%'],
        ['6', 'faithful', '93.55%'],
        ['7', 'grateful', '93.31%'],
        ['8', 'furious', '93.20%'],
        ['9', 'angry', '93.12%'],
        ['10', 'devastated', '92.88%'],
        ['11', 'terrified', '92.84%'],
    ],
    col_widths=[2, 5, 5]
)

p = doc.add_paragraph()
run = p.add_run('Finding: ')
run.bold = True
p.add_run('High similarity does not equal good emotion discrimination. The model makes all '
          'emotional text look alike in early layers.')

# 5.4
doc.add_heading('5.4 SCCA Results', level=2)

add_table(doc,
    ['Metric', 'Value'],
    [
        ['Canonical correlation range', '0.79 (Layer 0) to 0.84 (Layer 10-15)'],
        ['Within-class similarity', '~0.99 across all layers'],
        ['Between-class similarity', '~-0.09 (slightly negative)'],
        ['Discrimination gap', '~1.08 (very high, stable across layers)'],
        ['Best canonical correlation', 'Layer 10-11 (0.84)'],
    ],
    col_widths=[6, 11]
)

p = doc.add_paragraph()
run = p.add_run('Finding: ')
run.bold = True
p.add_run('The model maintains strong emotion-specific prompt-utterance alignment '
          'throughout the network. Comprehension genuinely flows into generation with '
          'emotion identity preserved. The negative between-class similarity indicates '
          'the model actively pushes different emotion categories apart in canonical space.')

# 5.5
doc.add_heading('5.5 Cross-Method Comparison', level=2)

add_table(doc,
    ['Method', 'Best Layers', 'Key Finding'],
    [
        ['Initial Probing (L2)', '5-9', 'Emotion is linearly decodable at 62% accuracy (6x chance)'],
        ['L1 Probing', '24-27', 'Emotion concentrates into ~40% of neurons in late layers'],
        ['Cosine Similarity', '2', 'Prompt/utterance representations are near-identical early (99.87%)'],
        ['Pairwise Matrices', '2', 'Alignment degrades from 99.87% to 80% across depth'],
        ['SCCA', '10-15', 'Emotion-specific comprehension-generation alignment is ~1.08 (strong)'],
    ],
    col_widths=[4, 3, 10]
)

doc.add_page_break()

# ============================================================
# 6. INTEGRATED INTERPRETATION
# ============================================================
doc.add_heading('6. Integrated Interpretation', level=1)

doc.add_paragraph(
    'The five analytical methods converge to reveal a coherent picture of how emotion '
    'is processed across the 29 transformer layers of Llama-3.2-3B:'
)

add_table(doc,
    ['Layer Range', 'Processing Phase', 'Evidence'],
    [
        ['Layers 0-2', 'Raw token processing. Everything looks similar. No emotion discrimination.',
         'Cosine similarity: 99.87%. Probing accuracy: 42%. Sparsity: 100%.'],
        ['Layers 3-9', 'Emotion boundaries form. Probing peaks. Distributed encoding across many neurons.',
         'Probing peaks at 62% (Layer 5). L2 outperforms L1. Cosine similarity begins dropping.'],
        ['Layers 10-15', 'Comprehension-generation alignment peaks. SCCA finds optimal shared structure.',
         'SCCA canonical correlation peaks at 0.84. SCCA gap stable at ~1.08.'],
        ['Layers 16-27', 'Emotion concentrates into specialized neurons. L1 catches up to L2.',
         'L1 peaks at 58.6% (Layer 26). Sparsity drops to 58%. Top features become identifiable.'],
        ['Layer 28', 'Output preparation. Emotion features are discarded.',
         'Sparsity spikes to 92.5%. Cosine similarity drops to 80%. Probing accuracy declines.'],
    ],
    col_widths=[3, 7, 7]
)

# ============================================================
# 7. CORE CONCLUSIONS
# ============================================================
doc.add_heading('7. Core Conclusions', level=1)

conclusions = [
    ('Llama-3.2-3B does encode emotion', 'not trivially, but with structured, layer-specific representations that are 6x above chance level.'),
    ('Positive emotions are much more separable than negative emotions', 'grateful and hopeful achieve 95% classification accuracy, while afraid and furious reach only 45-50%.'),
    ('Emotion encoding is distributed in mid-layers but concentrated in late layers', 'L2 probing peaks early (distributed signal), while L1 probing peaks late (sparse, localized neurons).'),
    ('The model\'s emotional understanding genuinely connects to its response generation', 'SCCA discrimination gap of ~1.08 across all layers demonstrates that comprehension faithfully flows into generation.'),
    ('A multi-layer approach is optimal', 'Layer 2 for semantic matching, Layers 5-15 for emotion classification, Layers 10-15 for comprehension-generation alignment.'),
    ('Dimension 243 is a candidate "emotional valence" neuron', 'appearing in the top-20 active features for 10 out of 11 emotions at the best-performing layer.'),
]

for i, (title, detail) in enumerate(conclusions, 1):
    p = doc.add_paragraph()
    run = p.add_run(f'{i}. {title}: ')
    run.bold = True
    p.add_run(detail)

doc.add_page_break()

# ============================================================
# 8. RELATED WORK
# ============================================================
doc.add_heading('8. Related Work & Annotations', level=1)

related_papers = [
    ('Lee et al. (2024) \u2014 IEEE ACII',
     'Looks at whether LLMs can generate responses that people perceive as empathetic. '
     'Participants compared supportive responses written by GPT-4, Turbo, Llama 2, Mistral, '
     'and humans across emotionally challenging scenarios. Humans judged the LLM responses as '
     'more empathetic than those written by humans. However, this study only evaluated the '
     'perceived quality of the responses and did not look at whether the models internally '
     'represent emotions during language processing. This paper supports Stage 3 of our '
     'methodology.'),
    ('EACL 2026 \u2014 Internal Emotion Representations',
     'Examined how LLMs internally represent emotion by analyzing hidden layer embeddings '
     'across multiple datasets and prompting strategies. Found LLMs develop consistent emotional '
     'representations shared across data sets, with the strongest emotion encoding occurring in '
     'later layers. Provides a strong foundation for Stage 1 of our methodology.'),
    ('Belinkov (2022) \u2014 Probing Classifiers',
     'A foundational paper on probing classifiers. Explains why researchers train simple '
     'classifiers on hidden embeddings and discusses strengths and limitations of probing methods. '
     'Justifies our probing classification methodology.'),
    ('Reimers & Gurevych (2019) \u2014 Sentence Embeddings',
     'Discusses sentence embedding and semantic similarity using cosine similarity.'),
    ('Tenney et al. (2019) \u2014 Layer-wise Analysis',
     'Explains that different layers learn different linguistic information, providing theoretical '
     'support for our layer-by-layer analysis of emotion representation.'),
    ('Ayers et al. (2023) \u2014 JAMA Internal Medicine',
     'Compared physician responses with ChatGPT responses to patient questions. ChatGPT answers '
     'were often preferred for both quality and empathy. Demonstrates why empathy matters in '
     'healthcare LLMs.'),
    ('Sharma et al. (2020) \u2014 EPITOME',
     'The EPITOME paper: emotional reactions, interpretations, and explorations framework.'),
    ('Li et al. (2023) \u2014 EMNLP',
     'Connects emotion perception with response generation. Supports Stage 2 of our methodology.'),
    ('Rashkin et al. (2019) \u2014 EmpatheticDialogues',
     'The empathy dataset used in our study.'),
    ('Saravia et al. (2018) \u2014 CARER',
     'CARER emotion dataset for emotion classification.'),
    ('Theory of Mind & Empathy as Multidimensional Constructs',
     'Explains how Theory of Mind and empathy are multidimensional processes supported by '
     'different brain systems. Distinguishes cognitive empathy (understanding how someone feels) '
     'from emotional/affective empathy (actually sharing that emotion). Key takeaway: LLMs most '
     'likely exhibit something closer to cognitive empathy rather than emotional empathy.'),
    ('Kosinski (2024) \u2014 Nature Human Behaviour',
     'Investigates whether LLMs truly possess Theory of Mind or generate human-like responses '
     'that mimic social reasoning. Found GPT-4 often matched or exceeded human performance while '
     'relying on different mechanisms.'),
    ('Survey: LLMs in Healthcare (2024)',
     'A survey of the current state of LLMs in healthcare, examining application, performance, '
     'and ethical/safety challenges.'),
]

for title, desc in related_papers:
    p = doc.add_paragraph()
    run = p.add_run(title + ': ')
    run.bold = True
    p.add_run(desc)
    p.paragraph_format.space_after = Pt(8)

doc.add_page_break()

# ============================================================
# 9. BIBLIOGRAPHY
# ============================================================
doc.add_heading('9. Bibliography', level=1)

references = [
    'Fitzpatrick, K. K., Darcy, A., & Vierhile, M. (2017). Delivering cognitive behavior therapy to young adults using a conversational agent. JMIR Mental Health, 4(2), e19.',
    'Gueorguieva, E., Lau, T., Hadjiandreou, E., & Ong, D. (2024). The language of an empathy-inducing narrative. In Proceedings of the Annual Meeting of the Cognitive Science Society (Vol. 46).',
    'Lee, Y. K., Suh, J., Zhan, H., Li, J. J., & Ong, D. C. (2024). Large language models produce responses perceived to be empathic. In 2024 12th International Conference on Affective Computing and Intelligent Interaction (ACII) (pp. 63-71). IEEE. DOI: 10.1109/ACII63134.2024.00012.',
    'Maheswaran, A., Chua, C., & Desarkar, M. S. (2025). Probing the inherent ability of large language models for generating empathetic responses. Indian Institute of Technology Hyderabad & Swinburne University of Technology.',
    'Rashkin, H., Smith, E. M., Li, M., & Boureau, Y.-L. (2019). Towards empathetic open-domain conversation models: A new benchmark and dataset. In Proceedings of the 57th Annual Meeting of the ACL (pp. 5370-5381).',
    'Riess, H. (2017). The science of empathy. Journal of Patient Experience, 4(2), 74-77. DOI: 10.1177/2374373517699267.',
    'Sharma, A., Lin, I. W., Miner, A. S., Atkins, D. C., & Althoff, T. (2021). Towards facilitating empathic conversations in online mental health support: A reinforcement learning approach. arXiv preprint arXiv:2101.07714.',
    'Xie, H. J., Zhang, J., Zhang, X., & Liu, K. (2024). Scoring with large language models: A study on measuring empathy of responses in dialogues. Portland State University.',
    'Zhang, Y., Radishian, C., Brunswicker, S., Whitenack, D., & Linna Jr., D. W. (2024). Empathetic language in LLMs under prompt engineering: A comparative study in the legal field. Procedia Computer Science / 6th International Conference on AI in Computational Linguistics (ACling 2024).',
    'Sharma, A., et al. (2020). A computational approach to understanding empathy expressed in text-based mental health support. In Proceedings of EMNLP 2020 (pp. 5263-5276).',
]

for i, ref in enumerate(references, 1):
    p = doc.add_paragraph(f'[{i}] {ref}')
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.left_indent = Cm(1)
    p.paragraph_format.first_line_indent = Cm(-1)

# ============================================================
# SAVE
# ============================================================
output_path = '/Volumes/Reading/Projects/Empathy-LLM/Empathy_LLM_Report.docx'
doc.save(output_path)
print(f"Saved: {output_path}")
