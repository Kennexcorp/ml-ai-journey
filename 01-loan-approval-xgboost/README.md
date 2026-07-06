# 01 · Loan Approval Prediction (XGBoost + SHAP)

Predict whether a loan application should be approved, learning from the
outcomes of previous applications. The model is a gradient-boosted decision-tree
ensemble (**XGBoost**); because a powerful ensemble is less transparent than a
linear model, its predictions are explained with **SHAP** (SHapley Additive
exPlanations) — combining state-of-the-art accuracy with post-hoc
interpretability.

## 🎯 Problem

Binary classification: given features of a loan applicant, predict `approved`
(True/False). The whole pipeline is organised as a single class,
`LoanApprovalModel`, whose methods run in sequence:

```
load_and_clean → build_features → train → evaluate → make_figures
```

Shared state (cleaned data, the train/test split, the fitted model and its
predictions) lives on the instance rather than being passed between free
functions.

## 🧠 Techniques & concepts

- **Gradient-boosted trees** with `XGBClassifier`
- **Class imbalance** handling via `scale_pos_weight`
- **Decision-threshold tuning** — optimising F1 instead of using the default 0.5
- **5-fold cross-validation** (ROC-AUC) for robust evaluation
- **Model explainability** with SHAP beeswarm plots
- **Permutation importance** for feature relevance
- Data cleaning, feature engineering, and fairness inspection

## 📊 Results

On a held-out test set of 2,500 applicants, with the decision threshold tuned to
0.30 (F1-optimal):

| Metric | Score |
|--------|:-----:|
| Accuracy | 0.656 |
| Precision | 0.632 |
| Recall | 0.984 |
| F1 | 0.770 |
| ROC-AUC | 0.703 |

5-fold CV ROC-AUC: **0.714 ± 0.005**. The low-threshold choice deliberately
favours recall (catching approvable applicants) at the cost of precision — a
trade-off worth revisiting depending on the business cost of each error type.

Generated figures live in [`Output/`](./Output):

| Figure | What it shows |
|--------|---------------|
| `xgb_fig1_data_quality.png` | Data quality / missingness overview |
| `xgb_fig2_relevance_fairness.png` | Feature relevance & fairness checks |
| `xgb_fig3_model_performance.png` | Performance (ROC, confusion matrix, etc.) |
| `xgb_fig4_shap_beeswarm.png` | SHAP beeswarm — per-feature impact on predictions |

## ▶️ Running it

With [`uv`](https://docs.astral.sh/uv/) (recommended — Python 3.12 is pinned):

```bash
uv sync
uv run python loan_approval.py            # uses Data/previousApplicants.csv
uv run python loan_approval.py path/to/other.csv
```

Without uv:

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib
python loan_approval.py
```

There is also an exploratory notebook:
[`xgboost_shap_loan_application_approval.ipynb`](./xgboost_shap_loan_application_approval.ipynb).

## 📁 Files

```
01-loan-approval-xgboost/
├── loan_approval.py     # the LoanApprovalModel pipeline
├── xgboost_shap_loan_application_approval.ipynb  # exploration notebook
├── Data/                # previousApplicants.csv
├── Output/              # generated figures
├── pyproject.toml       # uv project + pinned deps
└── uv.lock              # reproducible lockfile
```

## 💡 Lessons / ideas to revisit

- The recall-heavy threshold inflates approvals — model a real cost matrix and
  pick the threshold from expected cost, not F1 alone.
- ROC-AUC ~0.70 suggests the features have limited signal; try engineered
  interaction features or external data.
- Add calibration (`CalibratedClassifierCV`) so predicted probabilities are
  trustworthy, not just rank-ordered.
