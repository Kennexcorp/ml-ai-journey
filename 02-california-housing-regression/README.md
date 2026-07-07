# 02 · California Housing Regression

Predict the **median house value** of California census block groups from 1990
census features. A step up from a single-feature toy model into a realistic
multivariate regression workflow: missing values, a categorical feature, feature
engineering, and a proper preprocessing pipeline.

Dataset: [`housing.tgz`](https://github.com/ageron/data/raw/main/housing.tgz)
(20,640 rows), auto-downloaded into `Data/` by the notebook's `load_housing()`.

## 🎯 Problem

Supervised regression: given 9 predictors — `longitude`, `latitude`,
`housing_median_age`, `total_rooms`, `total_bedrooms`, `population`,
`households`, `median_income`, and the categorical `ocean_proximity` — predict
the continuous target `median_house_value`.

Two data-quality realities to handle: `total_bedrooms` has missing values, and
`ocean_proximity` must be encoded (one-hot).

## 🧠 Techniques & concepts

- Linear regression baseline vs. tree-based models (Decision Tree / Random Forest)
- Preprocessing pipelines: imputation, scaling, one-hot encoding
- Feature engineering (`rooms_per_household`, `bedrooms_per_room`, …)
- Stratified train/test split on income; cross-validated RMSE

## ▶️ Running it

Explore interactively in [`notebook.ipynb`](./notebook.ipynb); the clean,
reproducible version lives in `main.py`:

```bash
uv sync
uv run python main.py
```

## 💡 Lessons / ideas to revisit

- TODO
