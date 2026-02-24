"""
Probabilistic Calibration Module for CECI Framework.
Applies Platt scaling to produce well-calibrated probabilities.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import joblib
import os
from typing import Optional


class ProbabilityCalibrator:
    """
    Lightweight probabilistic calibration using Platt scaling
    (logistic regression on raw model outputs) to improve
    calibration and provide uncertainty estimates for high-risk cases.
    """

    def __init__(self):
        self.pid_calibrator = None
        self.peff_calibrator = None
        self.is_fitted = False

    def fit(
        self,
        pid_raw: np.ndarray,
        peff_raw: np.ndarray,
        pid_labels: np.ndarray,
        peff_labels: np.ndarray,
    ):
        """
        Fit Platt scaling calibrators on raw model outputs.

        Args:
            pid_raw: Raw PID probabilities from model
            peff_raw: Raw PEff probabilities from model
            pid_labels: Ground truth for PID
            peff_labels: Ground truth for PEff (derived from variance)
        """
        # Platt scaling for PID
        self.pid_calibrator = LogisticRegression(C=1.0, max_iter=1000)
        self.pid_calibrator.fit(pid_raw.reshape(-1, 1), pid_labels)

        # Platt scaling for PEff
        # Binarize PEff labels for calibration (threshold at 0.5)
        peff_binary = (peff_labels > 0.5).astype(int)
        # Only fit if we have both classes
        if len(np.unique(peff_binary)) > 1:
            self.peff_calibrator = LogisticRegression(C=1.0, max_iter=1000)
            self.peff_calibrator.fit(peff_raw.reshape(-1, 1), peff_binary)
        else:
            self.peff_calibrator = None

        self.is_fitted = True

    def calibrate(
        self, pid_raw: np.ndarray, peff_raw: np.ndarray,
    ) -> tuple:
        """
        Apply calibration to raw probabilities.

        Returns:
            (calibrated_pid, calibrated_peff, uncertainty_pid)
        """
        if not self.is_fitted:
            # If not fitted, return raw values with default uncertainty
            return pid_raw, peff_raw, np.full_like(pid_raw, 0.5)

        # Calibrated PID
        cal_pid = self.pid_calibrator.predict_proba(
            pid_raw.reshape(-1, 1)
        )[:, 1]

        # Calibrated PEff
        if self.peff_calibrator is not None:
            cal_peff = self.peff_calibrator.predict_proba(
                peff_raw.reshape(-1, 1)
            )[:, 1]
        else:
            cal_peff = peff_raw

        # Uncertainty estimate: distance from decision boundary
        # Higher uncertainty when probability is near 0.5
        uncertainty = 1.0 - 2.0 * np.abs(cal_pid - 0.5)

        return cal_pid, cal_peff, uncertainty

    def save(self, path: str):
        """Save calibrator to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "pid_calibrator": self.pid_calibrator,
            "peff_calibrator": self.peff_calibrator,
        }, path)

    @classmethod
    def load(cls, path: str) -> "ProbabilityCalibrator":
        """Load calibrator from disk."""
        data = joblib.load(path)
        instance = cls()
        instance.pid_calibrator = data["pid_calibrator"]
        instance.peff_calibrator = data["peff_calibrator"]
        instance.is_fitted = True
        return instance
