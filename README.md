# Look, It's a Fake!

[![Open Part I in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SpectreB/UB-kaggle-looks-its-a-fake/blob/main/part1/look_its_a_fake_part_1.ipynb)
[![Open Part II in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SpectreB/UB-kaggle-looks-its-a-fake/blob/main/part2/look_its_a_fake_part_2_clean.ipynb)

Binary classification of LLM-generated vs. real content — UB Master FDS Kaggle competition 2025/26.  
**Team**: Frenchies — Clara Bouvier, Bastien Laplace

| Part | Constraint | Leaderboard |
|---|---|---|
| Part I | LogisticRegression only | **3rd / 17** (tied 2nd) |
| Part II | No model constraint | **2nd / 17** |

---

## Problem

Two heterogeneous datasets, same binary label (`fake` / `real`):

- **Dataset A** (`derma`): tabular health/dermatology data — numeric and binary features with heavy missing values (~300 training rows).
- **Dataset B** (`text`): short news headlines (~5900 training rows, class-imbalanced).

---

## Repository structure

```
part1/
  look_its_a_fake_part_1.ipynb        # Part I — LogisticRegression only
part2/
  look_its_a_fake_part_2_clean.ipynb  # Part II — ensembles + augmentation
requirements.txt
```

---

## Approach

### Part I — Feature engineering under a hard model constraint

The only lever is the feature representation. Key decisions:

- **Dataset A**: drop leaky feature (`Doughnuts consumption`), logical imputation for mutually exclusive indicator groups, quantile-binned discretization of `Genetic Propensity` and `Skin X test` (lets a linear model approximate threshold effects), interaction features, missing-value indicators. `RobustScaler` for outlier resistance.
- **Dataset B**: four-view `ColumnTransformer` — word TF-IDF (1–4 grams), character CountVectorizer (2–4 grams), char_wb TF-IDF (3–5 grams), and hand-crafted style flags (capitalization, punctuation, clickbait/sensationalist keywords). `max_iter` tuned via 5-fold CV as the only available regularization knob with `penalty=None`.

### Part II — Ensembles and data augmentation

- **Dataset A**: soft-voting blend of CatBoostClassifier (Optuna-tuned) and the Part I LogisticRegression. Blend weights grid-searched on balanced accuracy to handle class imbalance. Optimal split: 0.98 CatBoost / 0.02 LogReg — CatBoost carries the prediction; it handles missing values and non-linear interactions natively without the discretization workarounds required for LogReg.
- **Dataset B**: triple vectorizer (same as Part I) + `StyleFeatureExtractor` (raw stylometric counts replacing keyword flags) + soft-voting ensemble of LogisticRegression, RandomForestClassifier, and LightGBM. Rule-based data augmentation generates synthetic fake headlines via lexical substitution to balance the training set to 1400 fake titles.

---

## Validation scores

| Dataset | Model | Score |
|---|---|---|
| A — Part I | LogisticRegression | 0.8083 (accuracy) |
| B — Part I | LogisticRegression + style features | 0.8538 (accuracy) |
| A — Part II | CatBoost alone | 0.8372 (balanced acc.) |
| A — Part II | Ensemble (CatBoost 0.98 + LogReg 0.02) | **0.8547** (balanced acc.) |
| B — Part II | LR + RF + LightGBM + augmentation | **0.9403** (accuracy) |

---

## Retrospective — what I'd do differently

### Dataset A

- **The 0.98/0.02 blend is a post-hoc signal.** The near-zero LogReg weight means CatBoost was the right model all along. The Part I preprocessing effort — discretization, interaction features, missing indicators — was compensating for the linear model's limitations, not capturing genuine domain structure. Running a quick SHAP analysis on an initial CatBoost fit *before* manual feature engineering would have made this obvious early.
- **A single 80/20 split is too noisy at ~300 rows.** Variance in the accuracy estimate is high enough that model selection based on a single split is unreliable. Nested CV or repeated stratified k-fold would give a more stable and honest performance signal.
- **Mean imputation discards correlation structure.** `IterativeImputer` (MICE) models each missing column as a function of the others. On a small, correlated tabular dataset like this one, it typically recovers more signal than per-column mean fill.

### Dataset B

- **Pre-trained embeddings would likely dominate.** A fine-tuned `DistilBERT` or `sentence-transformers` model operates in a semantic space where "COVID misinformation" and "pandemic conspiracy" are close — TF-IDF keeps them far apart. The n-gram approach is strong for short texts but has a hard ceiling.
- **Rule-based augmentation is brittle.** The substitution rules only reinforce vocabulary patterns already present in the training set. Using a small LLM (e.g. GPT-4o-mini) to generate synthetic fake headlines conditioned on style examples would have produced more diverse, realistic samples and a more robust classifier.
- **`max_iter` as regularization is a hack.** Tuning convergence iterations worked empirically but is conceptually fragile. With Part II's relaxed constraint, switching to `LogisticRegression(penalty='l1', solver='liblinear')` and tuning `C` directly would have been cleaner and more principled.

### General

- **No ablation study.** The README reports final scores but not the incremental contribution of each component. A systematic ablation (word TF-IDF alone → + char n-grams → + style features → + ensemble) would both sharpen the engineering narrative and make decisions reproducible.
- **Asymmetric hyperparameter tuning.** CatBoost was Optuna-tuned; RandomForest and LightGBM in the Dataset B ensemble were left at defaults. Tuning LightGBM — the most hyperparameter-sensitive of the three — would likely have pushed accuracy further.

---

## Setup

```bash
pip install -r requirements.txt
```

Notebooks are designed to run on Google Colab. Data files are loaded from Google Drive — update the paths in the data loading cells to match your own Drive structure.
