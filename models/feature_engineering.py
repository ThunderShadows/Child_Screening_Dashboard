"""
Feature Engineering Module for CECI Framework.
Computes session-level and longitudinal features from raw session data.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def compute_session_features(sessions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure session-level features exist. The synthetic generator already
    provides these, but this function standardizes the feature set and
    handles real data that may need derivation.

    Features per session:
        - accuracy (Equation 1: Accj = 1/Nj * Σ yij)
        - mean_reaction_time
        - hesitation_ratio
        - task_completion_rate
        - error_burst_rate
        - engagement_score
    """
    feature_cols = [
        "accuracy", "mean_reaction_time", "hesitation_ratio",
        "task_completion_rate", "error_burst_rate", "engagement_score",
    ]
    # Validate all features present
    for col in feature_cols:
        if col not in sessions_df.columns:
            raise ValueError(f"Missing session feature: {col}")
    return sessions_df


def compute_longitudinal_features(sessions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-child longitudinal (cross-session) features.

    Features computed:
        - mean_accuracy: μ_Acc across sessions
        - var_accuracy: Var(Acc) = 1/S * Σ(Accj - μ_Acc)² (Equation 3)
        - accuracy_trend: linear slope of accuracy over sessions
        - mean_rt: average reaction time across sessions
        - rt_trend: linear slope of reaction time over sessions
        - mean_hesitation: average hesitation ratio
        - hesitation_trend: slope of hesitation over sessions
        - mean_engagement: average engagement score
        - engagement_trend: slope of engagement over sessions
        - mean_error_burst: average error burst rate
        - mean_completion: average task completion rate
        - n_sessions: number of sessions
    """
    feature_rows = []

    for child_id, group in sessions_df.groupby("child_id"):
        group = group.sort_values("session")
        S = len(group)
        acc = group["accuracy"].values
        rt = group["mean_reaction_time"].values
        hes = group["hesitation_ratio"].values
        eng = group["engagement_score"].values

        # Mean and variance (Equation 3)
        mean_acc = np.mean(acc)
        var_acc = np.var(acc)  # population variance as in the paper

        # Linear trend slopes
        sessions_idx = np.arange(S)
        acc_trend = np.polyfit(sessions_idx, acc, 1)[0] if S > 1 else 0.0
        rt_trend = np.polyfit(sessions_idx, rt, 1)[0] if S > 1 else 0.0
        hes_trend = np.polyfit(sessions_idx, hes, 1)[0] if S > 1 else 0.0
        eng_trend = np.polyfit(sessions_idx, eng, 1)[0] if S > 1 else 0.0

        feature_rows.append({
            "child_id": child_id,
            "age_group": group["age_group"].iloc[0],
            "mean_accuracy": round(mean_acc, 4),
            "var_accuracy": round(var_acc, 4),
            "accuracy_trend": round(acc_trend, 4),
            "mean_rt": round(np.mean(rt), 4),
            "rt_trend": round(rt_trend, 4),
            "mean_hesitation": round(np.mean(hes), 4),
            "hesitation_trend": round(hes_trend, 4),
            "mean_engagement": round(np.mean(eng), 4),
            "engagement_trend": round(eng_trend, 4),
            "mean_error_burst": round(np.mean(group["error_burst_rate"].values), 4),
            "mean_completion": round(np.mean(group["task_completion_rate"].values), 4),
            "n_sessions": S,
        })

    return pd.DataFrame(feature_rows)


def prepare_sequence_data(
    sessions_df: pd.DataFrame,
    max_seq_len: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare ordered session sequences for temporal (LSTM) model.
    Each child → (seq_len, n_features) with zero-padding.

    Args:
        sessions_df: Session-level data
        max_seq_len: Maximum sequence length (pad/truncate)

    Returns:
        X_seq: (n_children, max_seq_len, n_features) array
        child_ids: (n_children,) array of child IDs
        seq_lengths: (n_children,) actual sequence lengths before padding
    """
    feature_cols = [
        "accuracy", "mean_reaction_time", "hesitation_ratio",
        "task_completion_rate", "error_burst_rate", "engagement_score",
    ]
    n_features = len(feature_cols)

    children = sessions_df["child_id"].unique()
    n_children = len(children)

    X_seq = np.zeros((n_children, max_seq_len, n_features), dtype=np.float32)
    seq_lengths = np.zeros(n_children, dtype=np.int32)
    child_ids = np.zeros(n_children, dtype=np.int32)

    for idx, child_id in enumerate(children):
        child_data = sessions_df[sessions_df["child_id"] == child_id].sort_values("session")
        feats = child_data[feature_cols].values.astype(np.float32)
        seq_len = min(len(feats), max_seq_len)

        X_seq[idx, :seq_len, :] = feats[:seq_len]
        seq_lengths[idx] = seq_len
        child_ids[idx] = child_id

    return X_seq, child_ids, seq_lengths


def get_feature_names() -> list:
    """Return feature names for the longitudinal model."""
    return [
        "mean_accuracy", "var_accuracy", "accuracy_trend",
        "mean_rt", "rt_trend", "mean_hesitation", "hesitation_trend",
        "mean_engagement", "engagement_trend",
        "mean_error_burst", "mean_completion", "n_sessions",
    ]
