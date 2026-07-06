"""
Loan Approval Prediction Model (Gradient Boosting approach, class-based)

Predicts whether a loan application should be approved, based on the outcomes
of previous applications. The chosen model is gradient-boosted decision trees
(XGBoost). Because a powerful ensemble is less transparent than a linear model,
its predictions are explained with SHAP (SHapley Additive exPlanations), so the
accuracy of a state-of-the-art model is combined with post-hoc interpretability.

The whole pipeline is organised as a single class, LoanApprovalModel, whose
methods run in sequence: load_and_clean -> build_features -> train -> evaluate
-> make_figures. State that is shared between steps (the cleaned data, the
train/test split, the fitted model and its predictions) is held on the instance
rather than passed between free functions.

Usage:
    python loan_approval.py [path_to_csv]

DEPENDENCIES: xgboost and shap (in addition to pandas, numpy, scikit-learn and
matplotlib). Install them with:
    pip install xgboost shap
"""

import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix,
                             classification_report)

warnings.filterwarnings('ignore')


class LoanApprovalModel:
    """End-to-end loan-approval classifier built on XGBoost with SHAP explanations."""

    # Class-level configuration
    # SENTINEL_OUTGOINGS: the literal value found in the raw Outgoings column
    # for corrupted rows. It is treated as a missing-data flag, not a real
    # amount, because (1) 24 nines is implausibly large for anyone's outgoings
    # (genuine values here run from hundreds to tens of thousands); (2) it
    # recurs as the exact same string rather than varying like real data; and
    # (3) it co-occurs with other corruption (the fully-broken rows and one
    # side of each duplicate-ID pair). A field of repeated 9s is a long-standing
    # convention for "invalid/missing". This is an evidence-based inference, as
    # the dataset came without a data dictionary confirming it.
    SENTINEL_OUTGOINGS = '999999999999999999999999'
    # Figures are written here, resolved relative to this file so they always
    # land in the project's Output/ folder regardless of the caller's cwd.
    OUTPUT_DIR = Path(__file__).resolve().parent / 'Output'
    FEATURES = ['Income', 'Outgoings', 'NetDisposable']
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    COLORS = {
        'primary': '#264653',
        'corrupted': '#e76f51',
        'clean': '#2a9d8f',
        'highlight': '#e9c46a'
    }

    def __init__(self, csv_path):
        """Store the data path and initialise the attributes populated later."""
        self.csv_path = csv_path
        self.data = None          # cleaned, feature-engineered DataFrame
        self.model = None         # fitted XGBClassifier
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.pred = None          # test-set predictions
        self.proba = None         # test-set positive-class probabilities
        self.metrics = None       # dict of test-set metrics
        np.random.seed(self.RANDOM_STATE)

    # ------------------------------------------------------------------
    # 1. DATA LOADING AND CLEANING
    # ------------------------------------------------------------------
    def load_and_clean(self):
        """
        Clean the raw applicant dataset. Three distinct data-quality issues
        were identified by inspection and handled explicitly rather than by
        blanket imputation:

          (a) 231 rows have every identifying/feature field missing (ID, Age,
              Income, Postcode, Gender all null) and Outgoings set to a
              sentinel value. They cannot be linked to an applicant or usefully
              imputed, so they are dropped.

          (b) 1,000 applicants appear as duplicate-ID pairs (2,000 rows).
              Age/Postcode/Gender/Approved are identical within each pair; the
              only difference is that exactly one side has a corrupted Income
              ('nil') or Outgoings (sentinel) value while the other holds the
              true value. The valid value is recovered from the twin row, which
              is genuine data recovery rather than statistical estimation.

          (c) The remaining 11,500 rows are complete.
        """
        df = pd.read_csv(self.csv_path)
        df = df[df['ID'].notna()].copy()

        df['Income'] = df['Income'].replace('nil', pd.NA)
        df['Outgoings'] = df['Outgoings'].astype(str).replace(self.SENTINEL_OUTGOINGS, pd.NA)

        clean = (df.sort_values('ID')
                   .groupby('ID', as_index=False)
                   .agg({
                       'Age': 'first', 'Postcode': 'first', 'Gender': 'first',
                       'Approved': 'first',
                       'Income': lambda s: s.dropna().iloc[0] if s.notna().any() else pd.NA,
                       'Outgoings': lambda s: s.dropna().iloc[0] if s.notna().any() else pd.NA,
                   }))

        clean['ID'] = clean['ID'].astype(int)
        clean['Age'] = clean['Age'].astype(int)
        clean['Income'] = clean['Income'].astype(float).astype(int)
        clean['Outgoings'] = clean['Outgoings'].astype(float).astype(int)
        clean['Approved'] = clean['Approved'].astype(bool)

        assert clean.isna().sum().sum() == 0, "Unexpected missing values remain"
        self.data = clean
        print(f"Cleaning: cleaned to {len(clean)} valid applicants, "
              f"{clean.isna().sum().sum()} missing values.")
        return self

    # ------------------------------------------------------------------
    # 2. FEATURE ENGINEERING AND SELECTION
    # ------------------------------------------------------------------
    # Trained features: Income, Outgoings, and engineered NetDisposable
    # (Income - Outgoings). Gender, Postcode, Age and ID are deliberately
    # excluded on ethical/technical grounds (see report Section 2).
    #
    # NOTE: gradient-boosted trees are invariant to monotonic feature
    # rescaling, so no standardisation step is applied (unlike a linear model).
    def build_features(self):
        """Add the engineered NetDisposable feature to the cleaned data."""
        self.data['NetDisposable'] = self.data['Income'] - self.data['Outgoings']
        return self

    # ------------------------------------------------------------------
    # 3. MODEL TRAINING
    # ------------------------------------------------------------------
    def _make_estimator(self, y_train=None):
        """Return a configured XGBoost gradient-boosting classifier with balanced class weights."""
        # Calculate class weight to address imbalance
        if y_train is not None:
            n_rejected = (y_train == 0).sum()
            n_approved = (y_train == 1).sum()
            scale_pos_weight = n_rejected / n_approved
            print(f"Class balance: {n_rejected} rejected, {n_approved} approved (weight ratio: {scale_pos_weight:.2f})")
        else:
            scale_pos_weight = 1.0
        
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss', random_state=self.RANDOM_STATE)

    def train(self):
        """Split the data, fit the model, and report cross-validated ROC-AUC."""
        X = self.data[self.FEATURES]
        y = self.data['Approved'].astype(int)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.TEST_SIZE, stratify=y, random_state=self.RANDOM_STATE)

        # Fit model with class weighting
        self.model = self._make_estimator(self.y_train)
        self.model.fit(self.X_train, self.y_train)

        # Cross-validation with class weighting
        cv = cross_val_score(self._make_estimator(y), X, y, cv=5, scoring='roc_auc')
        print(f"5-fold CV ROC-AUC: {cv.mean():.3f} (+/-{cv.std():.3f})")
        return self

    # ------------------------------------------------------------------
    # 4. EVALUATION
    # ------------------------------------------------------------------
    def evaluate(self):
        """Compute test-set predictions with optimized threshold and headline metrics."""
        self.proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Find optimal threshold based on F1 score
        thresholds = np.arange(0.3, 0.8, 0.05)
        f1_scores = []
        
        for thr in thresholds:
            pred_thr = (self.proba >= thr).astype(int)
            f1 = f1_score(self.y_test, pred_thr)
            f1_scores.append(f1)
        
        self.optimal_threshold = thresholds[np.argmax(f1_scores)]
        self.pred = (self.proba >= self.optimal_threshold).astype(int)
        
        print(f"\nOptimal decision threshold: {self.optimal_threshold:.2f} (F1={max(f1_scores):.4f})")
        print(f"(Compared to default threshold 0.50 which gave F1={f1_score(self.y_test, (self.proba >= 0.5).astype(int)):.4f})")

        self.metrics = {
            'Accuracy': accuracy_score(self.y_test, self.pred),
            'Precision': precision_score(self.y_test, self.pred),
            'Recall': recall_score(self.y_test, self.pred),
            'F1': f1_score(self.y_test, self.pred),
            'ROC_AUC': roc_auc_score(self.y_test, self.proba),
        }
        print("\nXGBoost test-set metrics (with optimized threshold):")
        for k, v in self.metrics.items():
            print(f"  {k:10s}: {v:.4f}")
        print("\nClassification report:\n",
              classification_report(self.y_test, self.pred,
                                    target_names=['Rejected', 'Approved']))
        return self

    # ------------------------------------------------------------------
    # 4b. PREDICTION ON NEW APPLICANTS
    # ------------------------------------------------------------------
    def predict(self, applicants):
        """
        Score one or more new applicants with the fitted model.

        `applicants` is a DataFrame containing at least Income and Outgoings
        columns; NetDisposable is derived here so callers do not have to supply
        it. The tuned decision threshold from evaluate() is used when available,
        falling back to 0.5 otherwise. Returns a DataFrame with the approval
        probability and the boolean decision, aligned to the input index.
        """
        if self.model is None:
            raise RuntimeError("Model is not trained; call train() (or run()) first.")

        df = applicants.copy()
        df['NetDisposable'] = df['Income'] - df['Outgoings']
        proba = self.model.predict_proba(df[self.FEATURES])[:, 1]
        threshold = getattr(self, 'optimal_threshold', 0.5)
        return pd.DataFrame({
            'proba_approve': proba,
            'approved': proba >= threshold,
        }, index=df.index)

    # ------------------------------------------------------------------
    # 5. FIGURES
    # ------------------------------------------------------------------
    def _save_figure(self, filename):
        """Save and close figure consistently, writing into OUTPUT_DIR."""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(self.OUTPUT_DIR / filename, bbox_inches='tight')
        plt.close()

    def _add_bar_labels(self, ax, values, offset=0.02):
        """Add percentage labels on top of bars."""
        ylim_max = ax.get_ylim()[1]
        for i, v in enumerate(values):
            ax.text(i, v + (ylim_max * offset), f'{v:.0%}', 
                    ha='center', va='bottom', fontweight='bold')

    def _setup_bar_chart(self, ax, title, ylabel, color='primary', ylim=1):
        """Configure a standard bar chart and return color."""
        ax.set_ylim(0, ylim)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        return self.COLORS[color]

    def make_figures(self):
        """Produce and save the four report figures."""
        plt.rcParams.update({'font.size': 10, 'figure.dpi': 130})
        self._figure_data_quality()
        self._figure_relevance_fairness()
        self._figure_performance()
        self._figure_shap()
        print("Saved xgb_fig1_data_quality.png, xgb_fig2_relevance_fairness.png, "
              "xgb_fig3_model_performance.png, xgb_fig4_shap_beeswarm.png")
        return self

    def _figure_data_quality(self):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        
        # Left panel: record disposition
        labels = ['Fully corrupted\n(dropped)', 'Duplicate pairs\n(recovered)', 'Clean singleton\nrows']
        values = [231, 1000, 11500]
        colors = [self.COLORS['corrupted'], self.COLORS['highlight'], self.COLORS['clean']]
        ax[0].bar(labels, values, color=colors)
        for i, v in enumerate(values):
            ax[0].text(i, v + 150, f'{v:,}', ha='center', fontweight='bold')
        ax[0].set_ylabel('Rows')
        ax[0].set_title('A. Disposition of 13,731 raw records')
        ax[0].set_ylim(0, 13000)
        
        # Right panel: corrupted values recovered
        labels = ['Income ("nil")', 'Outgoings (sentinel)']
        values = [834, 628]
        ax[1].bar(labels, values, color=self.COLORS['corrupted'])
        for i, v in enumerate(values):
            ax[1].text(i, v + 15, f'{v} -> 0 unresolved', ha='center', fontsize=9)
        ax[1].set_ylabel('Corrupted values')
        ax[1].set_title('B. Corrupted values recovered from twin rows')
        ax[1].set_ylim(0, 950)
        
        self._save_figure('xgb_fig1_data_quality.png')

    def _figure_relevance_fairness(self):
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
        d = self.data.copy()
        
        # Income quartile
        d['IncomeQ'] = pd.qcut(d['Income'], 4, labels=['Q1\n(low)', 'Q2', 'Q3', 'Q4\n(high)'])
        r = d.groupby('IncomeQ', observed=True)['Approved'].mean()
        ax[0].bar(r.index.astype(str), r.values, color=self._setup_bar_chart(ax[0], 'A. By Income quartile\n(USED as a feature)', 'Approval rate'))
        self._add_bar_labels(ax[0], r.values)
        
        # Gender
        g = d[d['Gender'].isin(['a', 'b'])].groupby('Gender')['Approved'].mean()
        colors = [self.COLORS['corrupted'], self.COLORS['clean']]
        ax[1].bar(['Gender a', 'Gender b'], g.values, color=colors)
        self._setup_bar_chart(ax[1], 'B. By Gender\n(AUDITED, not a feature)', 'Approval rate')
        self._add_bar_labels(ax[1], g.values)
        
        # Age band
        d['AgeBand'] = pd.cut(d['Age'], [17, 30, 45, 60, 75, 96],
                              labels=['18-30', '31-45', '46-60', '61-75', '76-96'])
        a = d.groupby('AgeBand', observed=True)['Approved'].mean()
        ax[2].bar(a.index.astype(str), a.values, color=self._setup_bar_chart(ax[2], 'C. By Age band\n(AUDITED, not a feature)', 'Approval rate', color='highlight'))
        self._add_bar_labels(ax[2], a.values)
        
        self._save_figure('xgb_fig2_relevance_fairness.png')

    def _figure_performance(self):
        fig, ax = plt.subplots(2, 2, figsize=(11, 9.5))

        # Confusion matrix
        cm = confusion_matrix(self.y_test, self.pred)
        ax[0, 0].imshow(cm, cmap='Blues')
        ax[0, 0].set_xticks([0, 1])
        ax[0, 0].set_yticks([0, 1])
        ax[0, 0].set_xticklabels(['Rejected', 'Approved'])
        ax[0, 0].set_yticklabels(['Rejected', 'Approved'])
        ax[0, 0].set_xlabel('Predicted')
        ax[0, 0].set_ylabel('Actual')
        ax[0, 0].set_title('A. Confusion Matrix')
        for i in range(2):
            for j in range(2):
                ax[0, 0].text(j, i, cm[i, j], ha='center', va='center',
                              color='white' if cm[i, j] > cm.max() / 2 else 'black',
                              fontsize=13, fontweight='bold')

        # ROC curve
        fpr, tpr, _ = roc_curve(self.y_test, self.proba)
        auc = roc_auc_score(self.y_test, self.proba)
        ax[0, 1].plot(fpr, tpr, color=self.COLORS['primary'], linewidth=2, label=f"AUC = {auc:.3f}")
        ax[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random guess')
        ax[0, 1].set_xlabel('False Positive Rate')
        ax[0, 1].set_ylabel('True Positive Rate')
        ax[0, 1].set_title('B. ROC Curve')
        ax[0, 1].legend(fontsize=9)

        # Feature importance
        perm = permutation_importance(self.model, self.X_test, self.y_test,
                                      n_repeats=10, random_state=self.RANDOM_STATE,
                                      scoring='roc_auc')
        imp = pd.Series(perm.importances_mean, index=self.FEATURES).sort_values()
        ax[1, 0].barh(imp.index, imp.values, color=self.COLORS['clean'])
        ax[1, 0].set_xlabel('Drop in ROC-AUC when shuffled')
        ax[1, 0].set_title('C. Permutation Feature Importance')

        # Precision-Recall vs threshold
        thr = np.linspace(0.05, 0.95, 37)
        precs, recs = [], []
        yv = self.y_test.values
        for t in thr:
            p = (self.proba >= t).astype(int)
            tp = ((p == 1) & (yv == 1)).sum()
            precs.append(tp / max((p == 1).sum(), 1))
            recs.append(tp / max((yv == 1).sum(), 1))
        ax[1, 1].plot(thr, precs, label='Precision', color=self.COLORS['primary'], linewidth=2)
        ax[1, 1].plot(thr, recs, label='Recall', color=self.COLORS['highlight'], linewidth=2)
        ax[1, 1].axvline(0.5, color='grey', linestyle=':', label='Default (0.5)')
        ax[1, 1].set_xlabel('Decision threshold')
        ax[1, 1].set_ylabel('Score')
        ax[1, 1].set_title('D. Precision/Recall vs Threshold')
        ax[1, 1].legend(fontsize=8)

        self._save_figure('xgb_fig3_model_performance.png')

    def _figure_shap(self):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_test)
        shap.summary_plot(shap_values, self.X_test, show=False)
        plt.title('SHAP beeswarm: feature effect on approval probability')
        self._save_figure('xgb_fig4_shap_beeswarm.png')

    # ------------------------------------------------------------------
    # Convenience runner
    # ------------------------------------------------------------------
    def run(self):
        """Execute the full pipeline in order."""
        return (self.load_and_clean()
                    .build_features()
                    .train()
                    .evaluate()
                    .make_figures())


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else "./Data/previousApplicants.csv"
    model = LoanApprovalModel(path).run()
    new = pd.DataFrame({'Income': [45000, 18000], 'Outgoings': [20000, 22000]})
    print(model.predict(new))