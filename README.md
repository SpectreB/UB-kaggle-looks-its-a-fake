# Look, It's a Fake!

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
  look_its_a_fake_part_1.ipynb      # Part I — LogisticRegression only
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

## Setup

```bash
pip install -r requirements.txt
```

Notebooks are designed to run on Google Colab. Data files are loaded from Google Drive — update the paths in the data loading cells to match your own Drive structure.
