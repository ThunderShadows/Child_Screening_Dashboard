"""
Tree-Based Model (XGBoost) for CECI Framework.
Interpretable baseline classifier for at-risk vs. typical detection.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix,
)
import joblib
import os
from typing import Dict, Tuple, Optional


class CECITreeModel:
    """
    XGBoost-based interpretable classifier for baseline risk prediction.
    Trained on aggregated longitudinal features.
    """

    def __init__(self, params: Optional[Dict] = None):
        self.default_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 5,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "use_label_encoder": False,
        }
        if params:
            self.default_params.update(params)

        self.model = xgb.XGBClassifier(**self.default_params)
        self.feature_names = None
        self.is_trained = False

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: list = None,
    ) -> Dict:
        """
        Train the XGBoost model with stratified cross-validation.

        Args:
            X: Feature matrix (n_children, n_features)
            y: Binary labels (0=typical, 1=at_risk)
            feature_names: Names of features for interpretability

        Returns:
            Dictionary of training metrics
        """
        self.feature_names = feature_names or list(X.columns) if isinstance(X, pd.DataFrame) else None

        # Cross-validation metrics
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_aucs = []
        cv_accs = []

        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = (
                X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx],
                X.iloc[val_idx] if isinstance(X, pd.DataFrame) else X[val_idx],
            )
            y_train, y_val = y[train_idx], y[val_idx]

            temp_model = xgb.XGBClassifier(**self.default_params)
            temp_model.fit(X_train, y_train, verbose=False)
            y_pred_proba = temp_model.predict_proba(X_val)[:, 1]
            cv_aucs.append(roc_auc_score(y_val, y_pred_proba))
            cv_accs.append(accuracy_score(y_val, (y_pred_proba > 0.5).astype(int)))

        # Final training on full data
        self.model.fit(X, y, verbose=False)
        self.is_trained = True

        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]

        metrics = {
            "cv_auc_mean": round(np.mean(cv_aucs), 4),
            "cv_auc_std": round(np.std(cv_aucs), 4),
            "cv_acc_mean": round(np.mean(cv_accs), 4),
            "train_auc": round(roc_auc_score(y, y_proba), 4),
            "train_accuracy": round(accuracy_score(y, y_pred), 4),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "classification_report": classification_report(y, y_pred, output_dict=True),
        }
        return metrics

    def predict_proba(self, X) -> np.ndarray:
        """Return probability of at-risk class."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X) -> np.ndarray:
        """Return binary predictions."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance scores."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        importance = self.model.feature_importances_
        if self.feature_names:
            return dict(sorted(
                zip(self.feature_names, importance),
                key=lambda x: x[1], reverse=True,
            ))
        return dict(enumerate(importance))

    def save(self, path: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "params": self.default_params,
        }, path)

    @classmethod
    def load(cls, path: str) -> "CECITreeModel":
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls(params=data["params"])
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        instance.is_trained = True
        return instance
