"""
CECI Inference Module.
Provides a simple API for the existing interface to call for predictions.
Load trained models and run end-to-end inference on new child session data.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.feature_engineering import (
    compute_longitudinal_features,
    prepare_sequence_data,
)
from models.tree_model import CECITreeModel
from models.temporal_model import TemporalModelTrainer
from models.calibration import ProbabilityCalibrator
from models.ceci_index import (
    compute_ceci,
    classify_risk_band,
    get_ceci_breakdown,
)
from data.synthetic_generator import AGE_GROUP_GAMES, generate_child_sessions


class CECIPredictor:
    """
    High-level inference API for the CECI screening system.
    Load this class in your existing interface to get predictions.

    Usage:
        predictor = CECIPredictor.load("saved_models")

        # From raw session data
        result = predictor.predict_from_sessions(sessions_list)

        # Generate a demo prediction
        result = predictor.predict_demo(age_group="3-5", profile="at_risk")
    """

    def __init__(
        self,
        tree_model: CECITreeModel,
        temporal_trainer: TemporalModelTrainer,
        calibrator: ProbabilityCalibrator,
    ):
        self.tree_model = tree_model
        self.temporal_trainer = temporal_trainer
        self.calibrator = calibrator

    @classmethod
    def load(cls, models_dir: str) -> "CECIPredictor":
        """
        Load all trained models from a directory.

        Args:
            models_dir: Path to the saved_models directory
        """
        tree_model = CECITreeModel.load(
            os.path.join(models_dir, "xgboost_model.joblib")
        )
        temporal_trainer = TemporalModelTrainer.load(
            os.path.join(models_dir, "lstm")
        )
        calibrator = ProbabilityCalibrator.load(
            os.path.join(models_dir, "calibrator.joblib")
        )
        return cls(tree_model, temporal_trainer, calibrator)

    def predict_from_sessions(
        self, sessions: List[Dict], child_id: int = 1, age_group: str = "3-5",
    ) -> Dict:
        """
        Run full CECI inference on a list of session feature dictionaries.

        Args:
            sessions: List of dicts, each with keys:
                accuracy, mean_reaction_time, hesitation_ratio,
                task_completion_rate, error_burst_rate, engagement_score
            child_id: Child identifier
            age_group: '0-2', '3-5', or '6-9'

        Returns:
            Full CECI breakdown dictionary
        """
        # Build sessions DataFrame
        for i, s in enumerate(sessions):
            s["child_id"] = child_id
            s["session"] = i + 1
            s["age_group"] = age_group

        sessions_df = pd.DataFrame(sessions)

        # Longitudinal features for tree model
        long_features = compute_longitudinal_features(sessions_df)
        from models.feature_engineering import get_feature_names
        feature_names = get_feature_names()
        X_tree = long_features[feature_names]

        # Tree model prediction (baseline)
        tree_proba = self.tree_model.predict_proba(X_tree)[0]

        # Sequence data for LSTM
        X_seq, child_ids, seq_lengths = prepare_sequence_data(sessions_df)
        pid_raw, peff_raw = self.temporal_trainer.predict(X_seq, seq_lengths)

        # Calibrate
        pid_cal, peff_cal, uncertainty = self.calibrator.calibrate(pid_raw, peff_raw)

        # Variance for CECI
        var_acc = long_features["var_accuracy"].values[0]
        # Normalize (use a reasonable max variance of 0.1)
        var_acc_norm = min(var_acc / 0.1, 1.0)

        # Full breakdown
        breakdown = get_ceci_breakdown(
            pid=float(pid_cal[0]),
            var_acc=float(var_acc_norm),
            peff=float(peff_cal[0]),
            uncertainty=float(uncertainty[0]),
        )

        # Add extra info
        breakdown["tree_baseline_proba"] = round(float(tree_proba), 4)
        breakdown["age_group"] = age_group
        breakdown["n_sessions"] = len(sessions)
        breakdown["session_accuracies"] = [
            round(s["accuracy"], 4) for s in sessions
        ]

        return breakdown

    def predict_demo(
        self,
        age_group: str = "3-5",
        profile: str = "typical",
        n_sessions: int = 7,
    ) -> Dict:
        """
        Generate a demo child and predict their CECI.

        Args:
            age_group: '0-2', '3-5', or '6-9'
            profile: 'typical', 'effort_variable', or 'at_risk'
            n_sessions: Number of sessions to simulate

        Returns:
            CECI breakdown + session data
        """
        sessions, meta = generate_child_sessions(
            child_id=1, age_group=age_group,
            profile=profile, n_sessions=n_sessions,
        )

        # Convert to simple feature dicts
        session_features = [
            {
                "accuracy": s["accuracy"],
                "mean_reaction_time": s["mean_reaction_time"],
                "hesitation_ratio": s["hesitation_ratio"],
                "task_completion_rate": s["task_completion_rate"],
                "error_burst_rate": s["error_burst_rate"],
                "engagement_score": s["engagement_score"],
            }
            for s in sessions
        ]

        result = self.predict_from_sessions(
            session_features, child_id=1, age_group=age_group,
        )
        result["profile"] = profile
        result["games_played"] = [s.get("games_played", []) for s in sessions]
        result["raw_sessions"] = sessions

        return result

    @staticmethod
    def get_available_games(age_group: str = None) -> Dict:
        """
        Get available games, optionally filtered by age group.

        Args:
            age_group: '0-2', '3-5', '6-9', or None for all

        Returns:
            Dictionary of games by age group
        """
        if age_group and age_group in AGE_GROUP_GAMES:
            info = AGE_GROUP_GAMES[age_group]
            return {
                age_group: {
                    "label": info["label"],
                    "games": [
                        {
                            "id": g["id"],
                            "name": g["name"],
                            "description": g["description"],
                            "cognitive_domain": g["cognitive_domain"],
                        }
                        for g in info["games"]
                    ],
                }
            }

        result = {}
        for ag, info in AGE_GROUP_GAMES.items():
            result[ag] = {
                "label": info["label"],
                "games": [
                    {
                        "id": g["id"],
                        "name": g["name"],
                        "description": g["description"],
                        "cognitive_domain": g["cognitive_domain"],
                    }
                    for g in info["games"]
                ],
            }
        return result


if __name__ == "__main__":
    import json

    print("Loading CECI Predictor...")
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
    predictor = CECIPredictor.load(models_dir)

    print("\n── Available Games ──")
    games = predictor.get_available_games()
    for ag, info in games.items():
        print(f"\n{info['label']}:")
        for g in info["games"]:
            print(f"  • {g['name']}: {g['description']} [{g['cognitive_domain']}]")

    print("\n── Demo Predictions ──")
    for profile in ["typical", "effort_variable", "at_risk"]:
        for age_group in ["0-2", "3-5", "6-9"]:
            result = predictor.predict_demo(age_group=age_group, profile=profile, n_sessions=7)
            print(
                f"\n[{profile.upper():20s}] Age {age_group} | "
                f"CECI: {result['ceci_score']:.4f} | "
                f"Band: {result['risk_band']:6s} | "
                f"{result['risk_label']}"
            )
