# 🧠 ML / AI Learning Journey

A running collection of machine-learning and AI projects I build to deepen my
understanding — one folder per project, each self-contained with its own README,
code, and notes. This page is the index.

> **How this repo is organised:** every project lives in a numbered folder
> (`01-`, `02-`, …) so it sorts chronologically. Each project folder has its own
> `README.md` explaining the problem, approach, and what I learned. Dependencies
> are isolated per project with [`uv`](https://docs.astral.sh/uv/).

## 📚 Projects

| # | Project | Concepts / Techniques | Status |
|---|---------|-----------------------|:------:|
| 01 | [Loan Approval Prediction](./01-loan-approval-xgboost) | XGBoost, SHAP explainability, feature engineering, class-based ML pipeline | ✅ |

**Status key:** ✅ complete · 🚧 in progress · 💡 planned

## 🗺️ Roadmap / ideas

Things I want to build next (unordered):

- [ ] Linear & logistic regression from scratch (gradient descent, no sklearn)
- [ ] A neural network from scratch (manual backprop / autograd)
- [ ] A CNN on an image dataset
- [ ] A transformer / attention mechanism from scratch
- [ ] An LLM-powered app (RAG, agents, etc.)

## 🛠️ Getting started

Each project is independent. To run one:

```bash
cd 01-loan-approval-xgboost
uv sync            # creates .venv and installs that project's deps
uv run python loan_approval.py
```

If you don't use `uv`, each project's README also lists a plain
`pip install` command.

## 📁 Repo conventions

- **One folder per project**, numbered for ordering.
- **Every project has a `README.md`** — problem, approach, results, lessons.
- **Data & model files are gitignored** by default (see `.gitignore`); small
  sample datasets are committed deliberately with `git add -f`.
- When a project outgrows "a learning exercise" and becomes a real tool, it
  graduates into its own dedicated repository.

---

*Maintained by Sylvester. New project ⇒ add a folder, a README, and one row to
the table above.*
