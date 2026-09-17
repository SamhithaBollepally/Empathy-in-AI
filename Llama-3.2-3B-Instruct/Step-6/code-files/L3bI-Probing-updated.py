# %%
"""Layer-wise L1 / L2 logistic probes on Set 1 prompt embeddings.

Fixes vs the previous version
-----------------------------
1. REGULARISATION IS TUNED. Before, C=1.0 was hard-coded for every layer; it is the
   single most important knob for a 3072-dim probe. Now the penalty strength is chosen
   per layer by stratified 5-fold CV on the training split.
2. LAYER SELECTION NO LONGER USES THE TEST SET. Before, 58 models were scored on the
   same 440 test samples and the max was reported - an optimistically biased number.
   Now the best layer is picked on CV accuracy, and the test split is touched once.
3. REAL L2. `weight_decay` in Adam is coupled and rescaled by Adam's adaptive
   denominator, so it was neither sklearn's C nor a clean ridge penalty. Now the ridge
   term is in the objective and solved to convergence with full-batch LBFGS.
4. REAL L1. Before, lambda=1.0 times a sum over 33k weights swamped the cross-entropy
   term, so the probe never learned (8-20% accuracy, i.e. chance), and Adam cannot
   produce exact zeros so the "sparsity" figure of ~49% was measuring noise. Now L1 is
   solved with FISTA (proximal gradient), which yields exact zeros and true sparsity.
5. DETERMINISTIC. Seeds fixed for numpy and torch.
6. ERROR STRUCTURE IS REPORTED, not just accuracy: macro-F1, confusion matrix, and
   accuracy over coarse emotion families (the label set contains near-synonyms such as
   afraid/terrified/anxious, which puts a hard ceiling on 10-way accuracy).
7. BASELINES ADDED. Accuracy means nothing without them: TF-IDF on raw text (how much
   is just lexical?), layer 0 (the static token-embedding floor), nearest-centroid, and
   a shuffled-label control that must land near chance.
8. LEARNING CURVE, to answer whether more prompts per emotion would actually help.
9. SEPARATION IS MEASURED WITH A FISHER RATIO, not average prototype cosine. Uncentred
   cosine was 0.96-1.00 at every layer (it measured the shared common component), and
   mean-centring does not rescue it: centring forces the prototypes to sum to zero,
   which pins the average pairwise cosine near -1/(K-1) regardless of geometry. Only the
   spread, the closest pair, and a between/within scatter ratio carry layer information.

Added after the first full run
------------------------------
10. WIDER PENALTY GRIDS. In run 1, CV accuracy was still rising at the largest L2 lambda
    and 25/29 layers chose a value at a grid edge, so the grid - not the representation -
    was capping accuracy (train 0.80 vs test 0.64 confirmed residual overfitting).
11. PAIRWISE SEPARABILITY for all 45 emotion pairs, each against a TF-IDF probe on the
    same pair. This distinguishes "the probe is weak" from "these two labels describe
    the same texts", which 10-way accuracy cannot do.
12. ACCURACY AT SEVERAL LABEL GRANULARITIES (see COLLAPSE_SCHEMES). The 10-way labelling
    remains the primary metric; the collapses are reported as diagnostics only.

Reads / writes Llama-3.2-3B-Instruct/Step-6/data/ only.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

# %%
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_PATH = '/Volumes/Reading/Projects/Empathy-LLM/Llama-3.2-3B-Instruct/'
# BASE_PATH = '/content/drive/MyDrive/Project-4/'
DATA_PATH = BASE_PATH + 'Step-6/data/'   # scripts live in Step-6/code-files/
os.makedirs(DATA_PATH, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

TEST_SIZE = 0.2
N_FOLDS = 5
# Penalty grids. These were widened after the first run, where CV accuracy was still
# rising at the largest L2 lambda and 25/29 layers selected a value at a grid edge -
# i.e. the grid, not the data, was capping accuracy. L1 collapses to chance by 1e-1,
# so its grid runs downward instead, with half-decade steps because L1 sparsity is
# very sensitive to lambda and a coarse grid made the per-layer sparsity jump around.
LAMBDAS_L2 = np.logspace(-4, 4, 9)     # ridge:  mean_CE + 0.5*lam*||W||_F^2
LAMBDAS_L1 = np.logspace(-7, -2, 11)   # lasso:  mean_CE + lam*||W||_1
LAMBDAS_PAIR = np.logspace(-1, 3, 5)   # coarser grid for the 45 binary pair probes

# Coarse families, used to check whether errors are confined to near-synonyms.
EMOTION_FAMILIES = {
    'afraid': 'fear', 'terrified': 'fear', 'anxious': 'fear',
    'angry': 'anger', 'furious': 'anger',
    'sad': 'sadness', 'devastated': 'sadness', 'lonely': 'sadness',
    'grateful': 'positive', 'hopeful': 'positive',
}

# The 10-way labelling stays PRIMARY: every headline number, the layer selection and the
# learning curve all use it, so results remain comparable with Steps 1-5. The collapses
# below are DIAGNOSTICS: 10-way accuracy alone conflates "the probe cannot read emotion"
# with "these two labels are not distinguishable in the text", and the collapses separate
# those. Binary pair probes (see the pair-separability section) put afraid/terrified and
# angry/furious at ~0.58 with the embeddings adding only ~0.01 over TF-IDF, i.e. no
# recoverable distinction; sad/devastated reached 0.72 with a real +0.07 embedding gain,
# so it is a genuine distinction and is deliberately NOT merged.
COLLAPSE_SCHEMES = {
    'as_labelled': {},
    'merge_indistinguishable_pairs': {'afraid': 'fear_acute', 'terrified': 'fear_acute',
                                      'angry': 'anger', 'furious': 'anger'},
    'families': EMOTION_FAMILIES,
}

# %%
data = np.load(DATA_PATH + 'prompt_embeddings.npz', allow_pickle=True)
X_all = data['X']                       # (num_layers, num_samples, hidden_dim)
y_all = data['labels']
texts_all = data['texts'].tolist()
EMOTIONS = data['emotions'].tolist()

NUM_LAYERS, NUM_SAMPLES, HIDDEN_DIM = X_all.shape
CHANCE = 1.0 / len(EMOTIONS)
print(f'X: {X_all.shape}  emotions: {len(EMOTIONS)}  chance: {CHANCE:.1%}')

label_map = {e: i for i, e in enumerate(EMOTIONS)}
y_idx = np.array([label_map[e] for e in y_all])

train_idx, test_idx = train_test_split(np.arange(NUM_SAMPLES), test_size=TEST_SIZE,
                                       random_state=SEED, stratify=y_idx)
y_train, y_test = y_idx[train_idx], y_idx[test_idx]
print(f'train: {len(train_idx)}  test: {len(test_idx)}')


# %%
def standardize(X_tr, X_te):
    """z-score using TRAIN statistics only (no leakage). Returns tensors + stats."""
    mu = X_tr.mean(dim=0, keepdim=True)
    sd = X_tr.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (X_tr - mu) / sd, (X_te - mu) / sd, mu, sd


def fit_l2(X, y, lam, num_classes, max_iter=200):
    """Ridge multinomial logistic regression, full-batch LBFGS to convergence.

    Objective: mean cross-entropy + 0.5 * lam * ||W||_F^2   (bias unpenalised)
    """
    W = torch.zeros(X.shape[1], num_classes, device=X.device, requires_grad=True)
    b = torch.zeros(num_classes, device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=max_iter, history_size=10,
                            tolerance_grad=1e-7, tolerance_change=1e-9,
                            line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X @ W + b, y) + 0.5 * lam * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def _spectral_norm(X, iters=30):
    """Largest singular value of X by power iteration (for the FISTA step size)."""
    v = torch.randn(X.shape[1], device=X.device)
    v /= v.norm()
    for _ in range(iters):
        v = X.T @ (X @ v)
        n = v.norm()
        if n == 0:
            return torch.tensor(1.0, device=X.device)
        v /= n
    return (X @ v).norm() / v.norm()


def fit_l1(X, y, lam, num_classes, max_iter=400, tol=1e-7):
    """Lasso multinomial logistic regression via FISTA (proximal gradient).

    Objective: mean cross-entropy + lam * ||W||_1   (bias unpenalised)
    Proximal soft-thresholding gives EXACT zeros, so sparsity is measurable.
    """
    n = X.shape[0]
    # Conservative Lipschitz bound for the smooth part (true bound is 0.5*s^2/n).
    step = 1.0 / (_spectral_norm(X) ** 2 / n).clamp(min=1e-12).item()

    W = torch.zeros(X.shape[1], num_classes, device=X.device)
    b = torch.zeros(num_classes, device=X.device)
    Z_W, Z_b, t_prev = W.clone(), b.clone(), 1.0
    prev_obj = float('inf')

    for it in range(max_iter):
        Zw = Z_W.detach().requires_grad_(True)
        Zb = Z_b.detach().requires_grad_(True)
        ce = torch.nn.functional.cross_entropy(X @ Zw + Zb, y)
        gW, gb = torch.autograd.grad(ce, [Zw, Zb])

        W_new = Z_W - step * gW
        W_new = torch.sign(W_new) * (W_new.abs() - step * lam).clamp(min=0.0)  # prox_L1
        b_new = Z_b - step * gb

        t = (1.0 + np.sqrt(1.0 + 4.0 * t_prev ** 2)) / 2.0
        mom = (t_prev - 1.0) / t
        Z_W = W_new + mom * (W_new - W)
        Z_b = b_new + mom * (b_new - b)
        W, b, t_prev = W_new, b_new, t

        if it % 25 == 0:
            with torch.no_grad():
                obj = (torch.nn.functional.cross_entropy(X @ W + b, y)
                       + lam * W.abs().sum()).item()
            if abs(prev_obj - obj) < tol * max(1.0, abs(prev_obj)):
                break
            prev_obj = obj

    return W, b


FITTERS = {'l1': fit_l1, 'l2': fit_l2}


def accuracy(W, b, X, y):
    with torch.no_grad():
        return ((X @ W + b).argmax(dim=1) == y).float().mean().item()


# %%
def cv_select_lambda(X_train_np, y_train, lambdas, penalty, num_classes):
    """Stratified k-fold CV over the penalty grid. Returns (best_lam, mean, std, curve)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(X_train_np, y_train))

    X_gpu = torch.tensor(X_train_np, device=device)
    y_gpu = torch.tensor(y_train, dtype=torch.long, device=device)

    curve = {}
    for lam in lambdas:
        scores = []
        for tr, va in folds:
            Xtr, Xva, _, _ = standardize(X_gpu[tr], X_gpu[va])
            W, b = FITTERS[penalty](Xtr, y_gpu[tr], float(lam), num_classes)
            scores.append(accuracy(W, b, Xva, y_gpu[va]))
        curve[float(lam)] = (float(np.mean(scores)), float(np.std(scores)))

    best_lam = max(curve, key=lambda k: curve[k][0])
    del X_gpu, y_gpu
    return best_lam, curve[best_lam][0], curve[best_lam][1], curve


def select_1se(curve):
    """1-SE rule: strongest penalty whose CV accuracy is still within 1 std of the best.

    For L1 this matters: the accuracy-optimal lambda is tiny and produces almost no
    zeros, so it cannot answer "which dimensions carry emotion?". The 1-SE lambda gives
    the sparsest probe that is not measurably worse, which is what top-feature analysis
    should be based on.
    """
    best = max(curve, key=lambda k: curve[k][0])
    floor = curve[best][0] - curve[best][1]
    return max([lam for lam in curve if curve[lam][0] >= floor])


def fit_final(X_train_np, X_test_np, y_train, y_test, lam, penalty, num_classes):
    """Refit on the full training split at the chosen lambda; evaluate on test."""
    Xtr = torch.tensor(X_train_np, device=device)
    Xte = torch.tensor(X_test_np, device=device)
    Xtr, Xte, mu, sd = standardize(Xtr, Xte)
    ytr = torch.tensor(y_train, dtype=torch.long, device=device)
    yte = torch.tensor(y_test, dtype=torch.long, device=device)

    W, b = FITTERS[penalty](Xtr, ytr, float(lam), num_classes)
    with torch.no_grad():
        preds = (Xte @ W + b).argmax(dim=1).cpu().numpy()

    return {
        'train_acc': accuracy(W, b, Xtr, ytr),
        'test_acc': accuracy(W, b, Xte, yte),
        'macro_f1': float(f1_score(y_test, preds, average='macro')),
        'sparsity': float((W == 0).float().mean().item()),
        'preds': preds,
        'W': W.cpu(), 'b': b.cpu(), 'mu': mu.cpu(), 'sd': sd.cpu(),
    }


def nearest_centroid_acc(X_train_np, X_test_np, y_train, y_test, num_classes):
    """Cheap non-parametric baseline: mean-centred cosine to per-class train centroids."""
    mean = X_train_np.mean(axis=0)
    Xtr, Xte = X_train_np - mean, X_test_np - mean
    cents = np.stack([Xtr[y_train == k].mean(axis=0) for k in range(num_classes)])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-12
    Xte_n = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-12)
    return float(((Xte_n @ cents.T).argmax(axis=1) == y_test).mean())


# %%
# --- Per-layer sweep: CV-tuned L1 and L2 probes ---
num_classes = len(EMOTIONS)
layer_results, layer_probes = {}, {}

print(f'\nSweeping {NUM_LAYERS} layers x 2 penalties with {N_FOLDS}-fold CV...')
for layer in range(NUM_LAYERS):
    X = X_all[layer]
    X_tr_np, X_te_np = X[train_idx], X[test_idx]
    entry = {'centroid_test_acc': nearest_centroid_acc(X_tr_np, X_te_np,
                                                       y_train, y_test, num_classes)}

    curves = {}
    for penalty, lambdas in (('l1', LAMBDAS_L1), ('l2', LAMBDAS_L2)):
        lam, cv_mean, cv_std, curve = cv_select_lambda(X_tr_np, y_train, lambdas,
                                                       penalty, num_classes)
        curves[penalty] = curve
        final = fit_final(X_tr_np, X_te_np, y_train, y_test, lam, penalty, num_classes)
        entry[penalty] = {
            'lambda': lam, 'cv_acc': cv_mean, 'cv_std': cv_std,
            'cv_curve': {str(k): v for k, v in curve.items()},
            'train_acc': final['train_acc'], 'test_acc': final['test_acc'],
            'macro_f1': final['macro_f1'], 'sparsity': final['sparsity'],
            'lambda_at_grid_edge': bool(lam in (lambdas[0], lambdas[-1])),
        }
        layer_probes.setdefault(layer, {})[penalty] = final

    # Sparse-but-equivalent L1 probe (1-SE rule) for top-feature analysis.
    lam_1se = select_1se(curves['l1'])
    sparse = fit_final(X_tr_np, X_te_np, y_train, y_test, lam_1se, 'l1', num_classes)
    entry['l1_1se'] = {'lambda': lam_1se, 'test_acc': sparse['test_acc'],
                       'macro_f1': sparse['macro_f1'], 'sparsity': sparse['sparsity']}
    layer_probes[layer]['l1_1se'] = sparse

    layer_results[layer] = entry
    print(f"  L{layer:2d} | L2 cv {entry['l2']['cv_acc']:.3f}+-{entry['l2']['cv_std']:.3f} "
          f"test {entry['l2']['test_acc']:.3f} lam {entry['l2']['lambda']:.1e} "
          f"| L1 cv {entry['l1']['cv_acc']:.3f} test {entry['l1']['test_acc']:.3f} "
          f"sp {entry['l1']['sparsity']:.1%} | centroid {entry['centroid_test_acc']:.3f}")

# %%
# Best layer chosen on CV accuracy (never on test) - see fix #2.
best_layer = {p: max(layer_results, key=lambda l: layer_results[l][p]['cv_acc'])
              for p in ('l1', 'l2')}
for p in ('l1', 'l2'):
    r = layer_results[best_layer[p]][p]
    print(f"\nBest {p.upper()} layer (by CV): {best_layer[p]}  "
          f"cv {r['cv_acc']:.4f}+-{r['cv_std']:.4f}  test {r['test_acc']:.4f}  "
          f"macro-F1 {r['macro_f1']:.4f}  lambda {r['lambda']:.1e}")

# Layers statistically indistinguishable from the best (CV mean within 1 CV std).
BL = best_layer['l2']
thresh = layer_results[BL]['l2']['cv_acc'] - layer_results[BL]['l2']['cv_std']
tied = [l for l in layer_results if layer_results[l]['l2']['cv_acc'] >= thresh]
print(f'Layers within 1 CV std of the best L2 layer: {tied}')

# %%
# --- Baseline: TF-IDF on raw text (how much of this is just lexical?) ---
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
train_texts = [texts_all[i] for i in train_idx]
test_texts = [texts_all[i] for i in test_idx]
Xt_tr = tfidf.fit_transform(train_texts)
Xt_te = tfidf.transform(test_texts)
tfidf_clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xt_tr, y_train)
tfidf_acc = float(tfidf_clf.score(Xt_te, y_test))
print(f'\nTF-IDF text baseline test acc: {tfidf_acc:.4f}')

# --- Control: shuffled labels must collapse to chance ---
rng = np.random.default_rng(SEED)
y_shuf = rng.permutation(y_train)
shuffled = fit_final(X_all[BL][train_idx], X_all[BL][test_idx], y_shuf, y_test,
                     layer_results[BL]['l2']['lambda'], 'l2', num_classes)
print(f"Shuffled-label control at layer {BL}: {shuffled['test_acc']:.4f} "
      f'(chance {CHANCE:.4f})')

# %%
# --- Error structure at the best L2 layer ---
preds_best = layer_probes[BL]['l2']['preds']
cm = confusion_matrix(y_test, preds_best, labels=range(num_classes))
per_emotion_recall = {EMOTIONS[k]: float(cm[k, k] / cm[k].sum()) for k in range(num_classes)}

def collapsed_accuracy(gmap):
    """Accuracy after merging labels per gmap. Isolates how much of the 10-way error is
    just near-synonym confusion rather than the probe failing to read emotion."""
    groups = [gmap.get(e, e) for e in EMOTIONS]
    names = sorted(set(groups))
    to_g = np.array([names.index(g) for g in groups])
    return float((to_g[preds_best] == to_g[y_test]).mean()), names


print(f'\nPer-emotion recall at layer {BL}:')
for e, r in sorted(per_emotion_recall.items(), key=lambda kv: kv[1]):
    print(f'  {e:12s} {r:.3f}')

granularity = {}
print('\nAccuracy by label granularity (as_labelled is the primary metric; '
      'the collapses are diagnostics):')
for scheme, gmap in COLLAPSE_SCHEMES.items():
    acc, names = collapsed_accuracy(gmap)
    granularity[scheme] = {'n_classes': len(names), 'accuracy': acc,
                           'chance': 1 / len(names), 'classes': names}
    print(f'  {scheme:32s} {len(names):2d}-way  acc {acc:.4f}  '
          f'(chance {1 / len(names):.3f})')
family_acc = granularity['families']['accuracy']
fam_names = granularity['families']['classes']

# Sensitivity: prompts that literally contain their own emotion word are trivially
# classifiable, so accuracy is reported with and without them.
leak_mask = np.array([EMOTIONS[y_idx[i]] in texts_all[i].lower() for i in test_idx])
subset_acc = lambda m: (float((preds_best[m] == y_test[m]).mean()) if m.any() else None)
leak_acc, clean_acc = subset_acc(leak_mask), subset_acc(~leak_mask)
print(f'Test acc on label-word-leaking prompts ({leak_mask.sum()}): {leak_acc}')
print(f'Test acc on non-leaking prompts ({(~leak_mask).sum()}): {clean_acc}')

# %%
# --- Top dimensions from the sparse (1-SE) L1 probe at the best L1 layer ---
BL1 = best_layer['l1']
r1se = layer_results[BL1]['l1_1se']
W_sparse = layer_probes[BL1]['l1_1se']['W'].numpy()   # (hidden_dim, num_classes)
print(f"\nSparse L1 probe at layer {BL1}: lambda {r1se['lambda']:.1e}  "
      f"test acc {r1se['test_acc']:.4f}  zeros {r1se['sparsity']:.1%}  "
      f"(vs dense L1 test acc {layer_results[BL1]['l1']['test_acc']:.4f})")

top_features = {}
for i, emotion in enumerate(EMOTIONS):
    w = W_sparse[:, i]
    nz = np.flatnonzero(w)
    order = nz[np.argsort(-np.abs(w[nz]))][:20]
    top_features[emotion] = {'dims': order.tolist(),
                             'weights': [float(w[d]) for d in order],
                             'n_nonzero': int(nz.size)}
shared_dims = sorted(set.intersection(*(set(v['dims']) for v in top_features.values()))) \
    if all(top_features[e]['dims'] for e in EMOTIONS) else []
print(f'Dimensions in the top-20 of every emotion (shared axes): {shared_dims}')

# %%
# --- Learning curve: would more prompts per emotion help? ---
# lambda is re-tuned at every training size, otherwise small subsets look artificially bad.
FRACTIONS = [0.25, 0.5, 0.75, 1.0]
learning_curve = {}
X_bl = X_all[BL]

for frac in FRACTIONS:
    if frac < 1.0:
        sub, _ = train_test_split(np.arange(len(train_idx)), train_size=frac,
                                  random_state=SEED, stratify=y_train)
    else:
        sub = np.arange(len(train_idx))
    Xs, ys = X_bl[train_idx[sub]], y_train[sub]
    lam, cv_mean, cv_std, _ = cv_select_lambda(Xs, ys, LAMBDAS_L2, 'l2', num_classes)
    final = fit_final(Xs, X_bl[test_idx], ys, y_test, lam, 'l2', num_classes)
    learning_curve[len(sub)] = {'test_acc': final['test_acc'], 'cv_acc': cv_mean,
                                'cv_std': cv_std, 'lambda': lam,
                                'per_emotion': len(sub) // num_classes}
    print(f'  n_train {len(sub):5d} ({len(sub) // num_classes:4d}/emotion) '
          f"cv {cv_mean:.4f} test {final['test_acc']:.4f}")

sizes = sorted(learning_curve)
gain_last = learning_curve[sizes[-1]]['test_acc'] - learning_curve[sizes[-2]]['test_acc']
print(f'Accuracy gained over the last 25% of training data: {gain_last:+.4f}')

# %%
# --- Pairwise separability: is a confusable pair distinguishable AT ALL? ---
# 10-way accuracy cannot tell "the probe is weak" from "these two labels describe the
# same texts". A binary probe on just the two classes can, and comparing it against a
# TF-IDF binary probe on the same pair shows whether the embeddings add anything beyond
# vocabulary. Run on the training split only; the test split stays untouched.
X_bl_train = X_all[BL][train_idx]
pair_sep = {}

print(f'\nBinary pair separability at layer {BL} (CV on train, chance 0.500):')
for i in range(num_classes):
    for j in range(i + 1, num_classes):
        m = (y_train == i) | (y_train == j)
        yb = (y_train[m] == j).astype(int)
        _, cv_emb, cv_std, _ = cv_select_lambda(X_bl_train[m], yb, LAMBDAS_PAIR, 'l2', 2)

        pair_texts = [train_texts[k] for k in np.flatnonzero(m)]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        cv_txt = float(cross_val_score(
            LogisticRegression(max_iter=3000), vec.fit_transform(pair_texts), yb,
            cv=StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)).mean())

        pair_sep[f'{EMOTIONS[i]}|{EMOTIONS[j]}'] = {
            'embedding_cv': cv_emb, 'embedding_cv_std': cv_std, 'tfidf_cv': cv_txt,
            'embedding_gain_over_tfidf': cv_emb - cv_txt}

ranked = sorted(pair_sep.items(), key=lambda kv: kv[1]['embedding_cv'])
print('  least separable pairs (candidates for merging):')
for name, v in ranked[:5]:
    print(f"    {name:26s} embed {v['embedding_cv']:.3f} tfidf {v['tfidf_cv']:.3f} "
          f"gain {v['embedding_gain_over_tfidf']:+.3f}")

# %%
# --- Prototype geometry ---
# NOTE: the AVERAGE off-diagonal cosine of mean-centred prototypes is useless as a
# separation metric: centring forces the K prototypes to sum to zero, which pins the
# average pairwise cosine near -1/(K-1) (= -0.111 for K=10) no matter the geometry. The
# first run duly returned -0.107..-0.110 at every single layer. Informative quantities
# are the SPREAD of the pairwise cosines, the closest pair, and a within-vs-between
# class scatter ratio, all of which do vary by layer.
ctx = np.load(DATA_PATH + 'context_embeddings.npz', allow_pickle=True)
protos, grand_mean = ctx['protos'], ctx['grand_mean']   # (layers, emotions, hidden)
off = ~np.eye(num_classes, dtype=bool)


def cosine_stats(M):
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    S = M @ M.T
    return {'mean': float(S[off].mean()), 'std': float(S[off].std()),
            'max': float(S[off].max())}, S


def fisher_ratio(A, y):
    """trace(between-class scatter) / trace(within-class scatter). Scale-free, so it is
    comparable across layers whose activations have very different magnitudes."""
    mu = A.mean(axis=0)
    between = sum(len(c) * ((c.mean(axis=0) - mu) ** 2).sum()
                  for c in (A[y == k] for k in range(num_classes)))
    within = sum(((c - c.mean(axis=0)) ** 2).sum()
                 for c in (A[y == k] for k in range(num_classes)))
    return float(between / within)


cos_raw, cos_centered, sim_centered, fisher = {}, {}, {}, {}
for layer in range(NUM_LAYERS):
    cos_raw[layer], _ = cosine_stats(protos[layer])
    cos_centered[layer], sim_centered[layer] = cosine_stats(protos[layer] - grand_mean[layer])
    fisher[layer] = fisher_ratio(X_all[layer], y_idx)

best_sep_layer = max(fisher, key=fisher.get)
print(f'\nMost separable layer by Fisher ratio: {best_sep_layer} '
      f'({fisher[best_sep_layer]:.4f}; layer 0 = {fisher[0]:.4f})')
print(f"  centred cosine mean at that layer {cos_centered[best_sep_layer]['mean']:+.4f} "
      f'(pinned near {-1 / (num_classes - 1):+.4f} - ignore it), '
      f"spread {cos_centered[best_sep_layer]['std']:.3f}, "
      f"closest pair cosine {cos_centered[best_sep_layer]['max']:.3f}")

# %%
# --- Persist probes: plain tensors, no pickled sklearn objects ---
probe_bundle = {
    'layers': {int(l): {p: {'W': layer_probes[l][p]['W'], 'b': layer_probes[l][p]['b'],
                            'mu': layer_probes[l][p]['mu'], 'sd': layer_probes[l][p]['sd'],
                            'lambda': layer_results[l][p]['lambda']}
                        for p in ('l1', 'l2', 'l1_1se')}
               for l in layer_probes},
    'classes': EMOTIONS,
    'best_layer_l1': int(best_layer['l1']),
    'best_layer_l2': int(best_layer['l2']),
    'hidden_dim': int(HIDDEN_DIM),
    'pooling': 'masked_mean_bos_excluded',
    'standardization': 'z-score with train mu/sd; apply (x-mu)/sd then x@W+b',
}
torch.save(probe_bundle, DATA_PATH + 'emotion_probes.pt')
print(f'\nSaved: {DATA_PATH}emotion_probes.pt')

# %%
results = {
    'config': {'emotions': EMOTIONS, 'chance': CHANCE, 'n_train': len(train_idx),
               'n_test': len(test_idx), 'num_layers': NUM_LAYERS,
               'hidden_dim': int(HIDDEN_DIM), 'n_folds': N_FOLDS, 'seed': SEED,
               'lambdas_l1': LAMBDAS_L1.tolist(), 'lambdas_l2': LAMBDAS_L2.tolist()},
    'per_layer': {str(l): layer_results[l] for l in layer_results},
    'best_layer_l1': int(best_layer['l1']),
    'best_layer_l2': int(best_layer['l2']),
    'layers_tied_with_best_l2': tied,
    'baselines': {'chance': CHANCE, 'tfidf_text': tfidf_acc,
                  'layer0_l2_test_acc': layer_results[0]['l2']['test_acc'],
                  'shuffled_label_control': shuffled['test_acc'],
                  'nearest_centroid_best_layer': layer_results[BL]['centroid_test_acc']},
    'best_layer_l2_detail': {
        'layer': int(BL),
        'test_acc': layer_results[BL]['l2']['test_acc'],
        'macro_f1': layer_results[BL]['l2']['macro_f1'],
        'per_emotion_recall': per_emotion_recall,
        'confusion_matrix': cm.tolist(),
        'family_accuracy': family_acc,
        'families': fam_names,
        'accuracy_by_granularity': granularity,
        'acc_label_word_leaking': leak_acc,
        'acc_non_leaking': clean_acc,
        'n_leaking_test': int(leak_mask.sum()),
    },
    'pair_separability': pair_sep,
    'sparse_l1_probe': {'layer': int(BL1), **r1se, 'top_features': top_features,
                        'dims_shared_across_all_emotions': shared_dims},
    'learning_curve': {str(k): v for k, v in learning_curve.items()},
    'prototype_cosine_raw': {str(k): v for k, v in cos_raw.items()},
    'prototype_cosine_centered': {str(k): v for k, v in cos_centered.items()},
    'fisher_ratio_per_layer': {str(k): v for k, v in fisher.items()},
    'best_separated_layer': int(best_sep_layer),
    'centred_cosine_mean_is_pinned_at': -1 / (num_classes - 1),
}
with open(DATA_PATH + 'probing_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved: {DATA_PATH}probing_results.json')

# %%
# Plot 1: CV accuracy per layer with fold spread, against every baseline.
layers = list(range(NUM_LAYERS))
fig, ax = plt.subplots(figsize=(13, 5))
for penalty, colour, label in (('l2', 'tab:red', 'L2 (ridge)'), ('l1', 'tab:blue', 'L1 (lasso)')):
    m = np.array([layer_results[l][penalty]['cv_acc'] for l in layers])
    s = np.array([layer_results[l][penalty]['cv_std'] for l in layers])
    ax.plot(layers, m, '-o', color=colour, markersize=4, label=label)
    ax.fill_between(layers, m - s, m + s, color=colour, alpha=0.15)

ax.plot(layers, [layer_results[l]['centroid_test_acc'] for l in layers], ':',
        color='tab:green', label='nearest centroid')
ax.axhline(tfidf_acc, color='tab:orange', ls='--', label=f'TF-IDF text ({tfidf_acc:.2f})')
ax.axhline(CHANCE, color='gray', ls=':', label=f'chance ({CHANCE:.1%})')
ax.axvline(BL, color='k', ls='--', alpha=0.4, label=f'best L2 layer ({BL})')
ax.set_xlabel('Layer')
ax.set_ylabel('Accuracy')
ax.set_title('CV-tuned probe accuracy per layer (shading = +-1 CV std across folds)')
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(DATA_PATH + 'probe_accuracy_per_layer.png', dpi=300)
print('Saved: probe_accuracy_per_layer.png')

# %%
# Plot 2: true L1 sparsity (exact zeros) and the CV-selected penalty per layer.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
ax1.plot(layers, [layer_results[l]['l1']['sparsity'] for l in layers], '-o',
         color='tab:blue', markersize=4, label='accuracy-optimal lambda')
ax1.plot(layers, [layer_results[l]['l1_1se']['sparsity'] for l in layers], '-s',
         color='tab:cyan', markersize=4, label='1-SE lambda (sparse, equivalent acc)')
ax1.set_ylabel('Fraction of exact zeros')
ax1.set_title('L1 sparsity (exact zeros from proximal solver)')
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

for penalty, colour in (('l1', 'tab:blue'), ('l2', 'tab:red')):
    ax2.semilogy(layers, [layer_results[l][penalty]['lambda'] for l in layers],
                 '-o', color=colour, markersize=4, label=penalty.upper())
ax2.set_xlabel('Layer')
ax2.set_ylabel('CV-selected lambda')
ax2.set_title('Selected penalty strength per layer (flat at a grid edge = widen the grid)')
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(DATA_PATH + 'l1_sparsity_and_lambda.png', dpi=300)
print('Saved: l1_sparsity_and_lambda.png')

# %%
# Plot 3: confusion matrix at the best L2 layer (row-normalised).
cm_norm = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(9, 7.5))
sns.heatmap(cm_norm, xticklabels=EMOTIONS, yticklabels=EMOTIONS, annot=True, fmt='.2f',
            cmap='Blues', vmin=0, vmax=1, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title(f'Confusion matrix, L2 probe at layer {BL} '
             f"(acc {layer_results[BL]['l2']['test_acc']:.3f}, "
             f'family acc {family_acc:.3f})')
plt.tight_layout()
plt.savefig(DATA_PATH + 'confusion_matrix_best_layer.png', dpi=300)
print('Saved: confusion_matrix_best_layer.png')

# %%
# Plot 4: learning curve - is accuracy still climbing at 400 prompts/emotion?
fig, ax = plt.subplots(figsize=(8, 5))
per_em = [learning_curve[n]['per_emotion'] for n in sizes]
ax.errorbar(per_em, [learning_curve[n]['cv_acc'] for n in sizes],
            yerr=[learning_curve[n]['cv_std'] for n in sizes],
            fmt='-o', capsize=3, label='CV accuracy')
ax.plot(per_em, [learning_curve[n]['test_acc'] for n in sizes], '--s', label='test accuracy')
ax.axhline(CHANCE, color='gray', ls=':', label='chance')
ax.set_xlabel('Training prompts per emotion')
ax.set_ylabel('Accuracy')
ax.set_title(f'Learning curve, L2 probe at layer {BL} (flat = more data will not help)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(DATA_PATH + 'learning_curve.png', dpi=300)
print('Saved: learning_curve.png')

# %%
# Plot 5: mean-centred prototype cosine heatmap at the best-separated layer.
fig, ax = plt.subplots(figsize=(9, 7.5))
sns.heatmap(sim_centered[best_sep_layer], xticklabels=EMOTIONS, yticklabels=EMOTIONS,
            annot=True, fmt='.2f', cmap='coolwarm', center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title(f'Mean-centred prototype cosine similarity, layer {best_sep_layer}\n'
             '(read the individual pairs; the average is pinned at -1/(K-1))')
plt.tight_layout()
plt.savefig(DATA_PATH + 'prototype_cosine_heatmap.png', dpi=300)
print('Saved: prototype_cosine_heatmap.png')

# %%
# Plot 6: separation per layer. Fisher ratio is the metric that actually varies; the two
# cosine means are drawn only to show why neither of them can be used.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax1.plot(layers, [fisher[l] for l in layers], '-o', color='tab:purple', markersize=4)
ax1.axvline(best_sep_layer, color='red', ls='--', alpha=0.6,
            label=f'most separable ({best_sep_layer})')
ax1.axvline(BL, color='k', ls=':', alpha=0.6, label=f'best probe layer ({BL})')
ax1.set_ylabel('between / within scatter')
ax1.set_title('Fisher separability per layer (higher = emotions more distinct)')
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

ax2.plot(layers, [cos_raw[l]['mean'] for l in layers], '-o', color='gray', markersize=4,
         label='raw mean cosine (dominated by shared component)')
ax2.plot(layers, [cos_centered[l]['mean'] for l in layers], '-o', color='tab:orange',
         markersize=4, label='centred mean cosine (pinned at -1/(K-1) - uninformative)')
ax2.plot(layers, [cos_centered[l]['max'] for l in layers], '-s', color='tab:green',
         markersize=4, label='centred cosine of the CLOSEST pair (informative)')
ax2.axhline(-1 / (num_classes - 1), color='tab:orange', ls=':', alpha=0.6)
ax2.set_xlabel('Layer')
ax2.set_ylabel('Cosine')
ax2.set_title('Why the average prototype cosine cannot be used as a separation metric')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(DATA_PATH + 'prototype_separation_per_layer.png', dpi=300)
print('Saved: prototype_separation_per_layer.png')

# %%
# Plot 7: pairwise separability - which emotion pairs are actually distinguishable, and
# how much the embeddings add over vocabulary alone.
fig, ax = plt.subplots(figsize=(9, 11))
names = [n for n, _ in ranked]
pos = np.arange(len(names))
ax.barh(pos, [pair_sep[n]['tfidf_cv'] for n in names], color='tab:orange', alpha=0.6,
        label='TF-IDF (vocabulary only)')
ax.barh(pos, [pair_sep[n]['embedding_cv'] for n in names], height=0.5, color='tab:blue',
        label=f'layer {BL} embeddings')
ax.axvline(0.5, color='gray', ls=':', label='chance')
ax.set_yticks(pos)
ax.set_yticklabels([n.replace('|', ' vs ') for n in names], fontsize=7)
ax.set_xlim(0.4, 1.0)
ax.set_xlabel('Binary CV accuracy')
ax.set_title(f'Pairwise emotion separability at layer {BL}\n'
             '(bars near 0.5 = the two labels are not distinguishable from the text)')
ax.legend(loc='lower right', fontsize=8)
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(DATA_PATH + 'pair_separability.png', dpi=300)
print('Saved: pair_separability.png')
