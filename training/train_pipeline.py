"""
End-to-End Training Pipeline for CECI ML Framework.
Generates synthetic data, trains all models, computes CECI, and saves artifacts.
"""

import sys
import os
import json
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic_generator import generate_dataset, AGE_GROUP_GAMES
from models.feature_engineering import (
    compute_longitudinal_features,
    prepare_sequence_data,
    get_feature_names,
)
from models.tree_model import CECITreeModel
from models.temporal_model import TemporalModelTrainer
from models.calibration import ProbabilityCalibrator
from models.ceci_index import (
    compute_ceci_batch,
    classify_risk_band_batch,
    get_ceci_breakdown,
)


SAVED_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_models"
)


def run_pipeline(n_children: int = 500, seed: int = 42):
    """Execute the full CECI training pipeline."""
    print("=" * 70)
    print("  CECI ML Training Pipeline")
    print("  Child Effort-Cognition Index for Early Screening")
    print("=" * 70)

    # ----------------------------------------------------------
    # Step 1: Generate Synthetic Data
    # ----------------------------------------------------------
    print("\n[1/7] Generating synthetic dataset...")
    sessions_df, children_df = generate_dataset(n_children, seed)
    print(f"  [OK] {len(children_df)} children, {len(sessions_df)} total sessions")
    print(f"  [OK] Profiles: {dict(children_df['profile'].value_counts())}")
    print(f"  [OK] Age groups: {dict(children_df['age_group'].value_counts())}")
    print(f"  [OK] Labels (at-risk=1): {dict(children_df['label'].value_counts())}")

    # Print available games per age group
    print("\n  Games by age group:")
    for ag, info in AGE_GROUP_GAMES.items():
        games = [g["name"] for g in info["games"]]
        print(f"    {info['label']}: {', '.join(games)}")

    # ----------------------------------------------------------
    # Step 2: Feature Engineering
    # ----------------------------------------------------------
    print("\n[2/7] Computing longitudinal features...")
    long_features = compute_longitudinal_features(sessions_df)

    # Merge labels
    long_features = long_features.merge(
        children_df[["child_id", "label", "profile"]], on="child_id",
    )
    print(f"  [OK] {len(long_features)} children x {len(get_feature_names())} features")
    print(f"  [OK] Sample features: {list(long_features.columns)}")

    # Prepare data for models
    feature_names = get_feature_names()
    X_tree = long_features[feature_names]
    y = long_features["label"].values

    # ----------------------------------------------------------
    # Step 3: Train XGBoost (Tree-Based Model)
    # ----------------------------------------------------------
    print("\n[3/7] Training XGBoost tree model...")
    tree_model = CECITreeModel()
    tree_metrics = tree_model.train(X_tree, y, feature_names)
    print(f"  [OK] CV AUC: {tree_metrics['cv_auc_mean']:.4f} +/- {tree_metrics['cv_auc_std']:.4f}")
    print(f"  [OK] CV Accuracy: {tree_metrics['cv_acc_mean']:.4f}")
    print(f"  [OK] Train AUC: {tree_metrics['train_auc']:.4f}")

    # Feature importance
    importance = tree_model.get_feature_importance()
    print("  [OK] Top 5 features:")
    for name, score in list(importance.items())[:5]:
        print(f"      {name}: {score:.4f}")

    # ----------------------------------------------------------
    # Step 4: Train LSTM (Temporal Model)
    # ----------------------------------------------------------
    print("\n[4/7] Training LSTM temporal model...")
    X_seq, child_ids, seq_lengths = prepare_sequence_data(sessions_df)

    # Align labels with sequence data order
    child_id_to_label = dict(zip(children_df["child_id"], children_df["label"]))
    child_id_to_var = dict(zip(long_features["child_id"], long_features["var_accuracy"]))
    seq_labels = np.array([child_id_to_label[cid] for cid in child_ids])
    seq_var_acc = np.array([child_id_to_var.get(cid, 0.0) for cid in child_ids])

    temporal_trainer = TemporalModelTrainer(
        input_size=6, hidden_size=64, num_layers=2, dropout=0.3, lr=0.001,
    )
    temporal_metrics = temporal_trainer.train(
        X_seq, seq_labels, seq_var_acc, seq_lengths, epochs=50, batch_size=32,
    )
    print(f"  [OK] Final Loss: {temporal_metrics['final_loss']:.4f}")
    print(f"  [OK] PID Mean: {temporal_metrics['pid_mean']:.4f}")
    print(f"  [OK] PEff Mean: {temporal_metrics['peff_mean']:.4f}")

    # ----------------------------------------------------------
    # Step 5: Get Predictions and Calibrate
    # ----------------------------------------------------------
    print("\n[5/7] Calibrating probabilities...")
    pid_raw, peff_raw = temporal_trainer.predict(X_seq, seq_lengths)

    # Align var_acc with the children order from prepare_sequence_data
    var_acc_aligned = seq_var_acc

    # Derive PEff labels for calibration
    var_norm = (seq_var_acc - seq_var_acc.min()) / (seq_var_acc.max() - seq_var_acc.min() + 1e-9)
    peff_labels = var_norm * (1 - seq_labels) * 0.8 + var_norm * seq_labels * 0.2

    calibrator = ProbabilityCalibrator()
    calibrator.fit(pid_raw, peff_raw, seq_labels, peff_labels)
    pid_cal, peff_cal, uncertainty = calibrator.calibrate(pid_raw, peff_raw)
    print(f"  [OK] Calibrated PID range: [{pid_cal.min():.4f}, {pid_cal.max():.4f}]")
    print(f"  [OK] Calibrated PEff range: [{peff_cal.min():.4f}, {peff_cal.max():.4f}]")

    # ----------------------------------------------------------
    # Step 6: Compute CECI Index
    # ----------------------------------------------------------
    print("\n[6/7] Computing CECI scores...")

    # Normalize var_acc to [0,1] for CECI formula
    var_acc_norm = var_acc_aligned / (var_acc_aligned.max() + 1e-9)

    ceci_scores = compute_ceci_batch(pid_cal, var_acc_norm, peff_cal)
    risk_bands = classify_risk_band_batch(ceci_scores)

    band_counts = {}
    for rb in risk_bands:
        band_counts[rb["band"]] = band_counts.get(rb["band"], 0) + 1
    print(f"  [OK] CECI range: [{ceci_scores.min():.4f}, {ceci_scores.max():.4f}]")
    print(f"  [OK] Mean CECI: {ceci_scores.mean():.4f}")
    print(f"  [OK] Risk distribution: {band_counts}")

    # Print sample breakdowns
    print("\n  Sample CECI Breakdowns:")
    for profile_type in ["typical", "effort_variable", "at_risk"]:
        # Find a child for this profile
        profile_children = children_df[children_df["profile"] == profile_type]
        if len(profile_children) > 0:
            sample_cid = profile_children.iloc[0]["child_id"]
            idx = np.where(child_ids == sample_cid)[0]
            if len(idx) > 0:
                i = idx[0]
                breakdown = get_ceci_breakdown(
                    pid_cal[i], var_acc_norm[i], peff_cal[i], uncertainty[i],
                )
                print(f"\n  [{profile_type.upper()}] Child {sample_cid}:")
                print(f"    CECI: {breakdown['ceci_score']:.4f} ({breakdown['risk_label']})")
                print(f"    PID: {breakdown['components']['pid']['value']:.4f} -> {breakdown['components']['pid']['interpretation']}")
                print(f"    Consistency: {breakdown['components']['consistency']['value']:.4f} -> {breakdown['components']['consistency']['interpretation']}")
                print(f"    Effort: {breakdown['components']['effort']['value']:.4f} -> {breakdown['components']['effort']['interpretation']}")
                print(f"    Clinical: {breakdown['clinical_note']}")

    # ----------------------------------------------------------
    # Step 7: Save All Model Artifacts
    # ----------------------------------------------------------
    print(f"\n[7/7] Saving models to {SAVED_MODELS_DIR}...")
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    tree_model.save(os.path.join(SAVED_MODELS_DIR, "xgboost_model.joblib"))
    temporal_trainer.save(os.path.join(SAVED_MODELS_DIR, "lstm"))
    calibrator.save(os.path.join(SAVED_MODELS_DIR, "calibrator.joblib"))

    # Save metadata
    metadata = {
        "n_children": n_children,
        "seed": seed,
        "feature_names": feature_names,
        "tree_metrics": {
            "cv_auc": tree_metrics["cv_auc_mean"],
            "cv_accuracy": tree_metrics["cv_acc_mean"],
        },
        "temporal_metrics": {
            "final_loss": temporal_metrics["final_loss"],
            "pid_mean": temporal_metrics["pid_mean"],
            "peff_mean": temporal_metrics["peff_mean"],
        },
        "ceci_stats": {
            "mean": round(float(ceci_scores.mean()), 4),
            "std": round(float(ceci_scores.std()), 4),
            "risk_distribution": band_counts,
        },
        "feature_importance": {k: round(float(v), 4) for k, v in importance.items()},
        "games": {
            ag: [g["name"] for g in info["games"]]
            for ag, info in AGE_GROUP_GAMES.items()
        },
    }
    with open(os.path.join(SAVED_MODELS_DIR, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  [OK] XGBoost model saved")
    print(f"  [OK] LSTM model saved")
    print(f"  [OK] Calibrator saved")
    print(f"  [OK] Training metadata saved")

    print("\n" + "=" * 70)
    print("  Pipeline Complete [OK]")
    print("=" * 70)

    return {
        "tree_model": tree_model,
        "temporal_trainer": temporal_trainer,
        "calibrator": calibrator,
        "metadata": metadata,
        "sessions_df": sessions_df,
        "children_df": children_df,
        "ceci_scores": ceci_scores,
        "risk_bands": risk_bands,
    }


if __name__ == "__main__":
    results = run_pipeline(n_children=500, seed=42)
