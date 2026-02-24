"""
Synthetic Data Generator for CECI Screening Framework.
Generates realistic multi-session child interaction data with
age-group-specific games and three behavioral profiles.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# ──────────────────────────────────────────────────────────────
# Age-Group Specific Game Definitions
# ──────────────────────────────────────────────────────────────

AGE_GROUP_GAMES = {
    "0-2": {
        "label": "Infant/Toddler (0–2 years)",
        "games": [
            {
                "id": "tap_the_animal",
                "name": "Tap the Animal",
                "description": "Tap on the animal that makes the sound played",
                "cognitive_domain": "auditory_recognition",
                "trials_range": (8, 15),
                "base_reaction_time": 3.0,
            },
            {
                "id": "peek_a_boo_memory",
                "name": "Peek-a-Boo Memory",
                "description": "Remember where the toy was hidden under the cup",
                "cognitive_domain": "working_memory",
                "trials_range": (6, 10),
                "base_reaction_time": 4.0,
            },
            {
                "id": "shape_sorter",
                "name": "Shape Sorter",
                "description": "Drag the shape into the matching hole",
                "cognitive_domain": "visual_spatial",
                "trials_range": (6, 12),
                "base_reaction_time": 4.5,
            },
        ],
    },
    "3-5": {
        "label": "Preschool (3–5 years)",
        "games": [
            {
                "id": "color_pattern_match",
                "name": "Color Pattern Match",
                "description": "Complete the color pattern sequence",
                "cognitive_domain": "pattern_recognition",
                "trials_range": (10, 20),
                "base_reaction_time": 2.5,
            },
            {
                "id": "memory_card_flip",
                "name": "Memory Card Flip",
                "description": "Find matching pairs of cards",
                "cognitive_domain": "working_memory",
                "trials_range": (10, 16),
                "base_reaction_time": 2.0,
            },
            {
                "id": "counting_garden",
                "name": "Counting Garden",
                "description": "Count the flowers and tap the correct number",
                "cognitive_domain": "numerical_cognition",
                "trials_range": (10, 20),
                "base_reaction_time": 2.5,
            },
            {
                "id": "story_sequence",
                "name": "Story Sequence",
                "description": "Arrange picture cards to tell the story in order",
                "cognitive_domain": "sequential_reasoning",
                "trials_range": (8, 14),
                "base_reaction_time": 3.0,
            },
        ],
    },
    "6-9": {
        "label": "Early School (6–9 years)",
        "games": [
            {
                "id": "digit_span_recall",
                "name": "Digit Span Recall",
                "description": "Remember and repeat the number sequence",
                "cognitive_domain": "working_memory",
                "trials_range": (12, 25),
                "base_reaction_time": 1.8,
            },
            {
                "id": "word_category_sort",
                "name": "Word Category Sort",
                "description": "Sort words into the correct category",
                "cognitive_domain": "verbal_reasoning",
                "trials_range": (15, 25),
                "base_reaction_time": 1.5,
            },
            {
                "id": "spatial_puzzle",
                "name": "Spatial Puzzle",
                "description": "Rotate and fit the piece into the puzzle",
                "cognitive_domain": "visual_spatial",
                "trials_range": (10, 20),
                "base_reaction_time": 2.0,
            },
            {
                "id": "math_race",
                "name": "Math Race",
                "description": "Solve arithmetic problems as quickly as possible",
                "cognitive_domain": "numerical_cognition",
                "trials_range": (15, 30),
                "base_reaction_time": 1.5,
            },
            {
                "id": "go_nogo",
                "name": "Go / No-Go",
                "description": "Tap for green shapes, hold back for red shapes",
                "cognitive_domain": "inhibitory_control",
                "trials_range": (20, 30),
                "base_reaction_time": 0.8,
            },
        ],
    },
}


# ──────────────────────────────────────────────────────────────
# Child Profile Generators
# ──────────────────────────────────────────────────────────────

def _generate_trial_data(
    n_trials: int,
    accuracy_mean: float,
    rt_base: float,
    rt_noise: float,
    hesitation_prob: float,
    profile: str,
) -> pd.DataFrame:
    """Generate individual trial-level data for one game in one session."""
    correct = np.random.binomial(1, accuracy_mean, n_trials)
    reaction_times = np.clip(
        np.random.lognormal(np.log(rt_base), rt_noise, n_trials), 0.2, 15.0
    )
    # Incorrect trials tend to have longer reaction times
    reaction_times[correct == 0] *= np.random.uniform(1.2, 1.8, np.sum(correct == 0))
    hesitations = np.random.binomial(1, hesitation_prob, n_trials)
    return pd.DataFrame({
        "trial": np.arange(1, n_trials + 1),
        "correct": correct,
        "reaction_time": np.round(reaction_times, 3),
        "hesitation": hesitations,
    })


def _session_features_from_trials(trials_df: pd.DataFrame, game_id: str) -> Dict:
    """Compute session-level features from trial data (Equation 1)."""
    n = len(trials_df)
    accuracy = trials_df["correct"].mean()
    mean_rt = trials_df["reaction_time"].mean()
    hesitation_ratio = trials_df["hesitation"].mean()
    completed = trials_df["correct"].sum()
    task_completion_rate = completed / n if n > 0 else 0.0

    # Error burst = 3+ consecutive errors
    error_runs = []
    run_len = 0
    for c in trials_df["correct"]:
        if c == 0:
            run_len += 1
        else:
            if run_len >= 3:
                error_runs.append(run_len)
            run_len = 0
    if run_len >= 3:
        error_runs.append(run_len)
    error_burst_rate = len(error_runs) / max(1, n // 3)

    # Engagement: derived from RT consistency + completion
    rt_cv = trials_df["reaction_time"].std() / (mean_rt + 1e-9)
    engagement_score = np.clip(1.0 - rt_cv * 0.5 - hesitation_ratio * 0.3, 0, 1)

    return {
        "game_id": game_id,
        "accuracy": round(accuracy, 4),
        "mean_reaction_time": round(mean_rt, 4),
        "hesitation_ratio": round(hesitation_ratio, 4),
        "task_completion_rate": round(task_completion_rate, 4),
        "error_burst_rate": round(error_burst_rate, 4),
        "engagement_score": round(engagement_score, 4),
        "n_trials": n,
    }


def generate_child_sessions(
    child_id: int,
    age_group: str,
    profile: str,
    n_sessions: int = None,
) -> Tuple[List[Dict], Dict]:
    """
    Generate full multi-session data for one child.

    Args:
        child_id: Unique child identifier
        age_group: '0-2', '3-5', or '6-9'
        profile: 'typical', 'effort_variable', or 'at_risk'
        n_sessions: Number of sessions (random 5-10 if None)

    Returns:
        (list of session feature dicts, child metadata dict)
    """
    if n_sessions is None:
        n_sessions = np.random.randint(5, 11)

    games = AGE_GROUP_GAMES[age_group]["games"]

    # Profile-specific parameters
    profiles = {
        "typical": {
            "acc_base": 0.82, "acc_var": 0.05,
            "rt_noise": 0.25, "hesitation": 0.08, "label": 0,
        },
        "effort_variable": {
            "acc_base": 0.55, "acc_var": 0.20,
            "rt_noise": 0.45, "hesitation": 0.25, "label": 0,
        },
        "at_risk": {
            "acc_base": 0.35, "acc_var": 0.06,
            "rt_noise": 0.30, "hesitation": 0.15, "label": 1,
        },
    }
    p = profiles[profile]

    sessions = []
    for s in range(1, n_sessions + 1):
        # Per-session accuracy varies by profile
        session_acc = np.clip(
            np.random.normal(p["acc_base"], p["acc_var"]), 0.05, 0.98
        )
        # Pick 2-3 games per session
        n_games = min(len(games), np.random.randint(2, 4))
        chosen_games = np.random.choice(games, n_games, replace=False)

        session_game_features = []
        for game in chosen_games:
            n_trials = np.random.randint(*game["trials_range"])
            trials = _generate_trial_data(
                n_trials, session_acc, game["base_reaction_time"],
                p["rt_noise"], p["hesitation"], profile,
            )
            feats = _session_features_from_trials(trials, game["id"])
            session_game_features.append(feats)

        # Aggregate across games in this session
        avg_features = {
            "child_id": child_id,
            "session": s,
            "age_group": age_group,
            "accuracy": round(np.mean([f["accuracy"] for f in session_game_features]), 4),
            "mean_reaction_time": round(np.mean([f["mean_reaction_time"] for f in session_game_features]), 4),
            "hesitation_ratio": round(np.mean([f["hesitation_ratio"] for f in session_game_features]), 4),
            "task_completion_rate": round(np.mean([f["task_completion_rate"] for f in session_game_features]), 4),
            "error_burst_rate": round(np.mean([f["error_burst_rate"] for f in session_game_features]), 4),
            "engagement_score": round(np.mean([f["engagement_score"] for f in session_game_features]), 4),
            "games_played": [f["game_id"] for f in session_game_features],
        }
        sessions.append(avg_features)

    child_meta = {
        "child_id": child_id,
        "age_group": age_group,
        "age_group_label": AGE_GROUP_GAMES[age_group]["label"],
        "profile": profile,
        "label": p["label"],  # 1=at_risk, 0=not at_risk
        "n_sessions": n_sessions,
    }
    return sessions, child_meta


def generate_dataset(
    n_children: int = 500,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a full synthetic dataset.

    Returns:
        (sessions_df, children_df)
    """
    np.random.seed(seed)

    all_sessions = []
    all_children = []
    age_groups = ["0-2", "3-5", "6-9"]

    # Distribution: 50% typical, 25% effort_variable, 25% at_risk
    profiles = (
        ["typical"] * int(n_children * 0.50)
        + ["effort_variable"] * int(n_children * 0.25)
        + ["at_risk"] * int(n_children * 0.25)
    )
    # Pad if rounding caused mismatch
    while len(profiles) < n_children:
        profiles.append("typical")
    np.random.shuffle(profiles)

    for cid in range(1, n_children + 1):
        age_group = np.random.choice(age_groups)
        profile = profiles[cid - 1]
        sessions, meta = generate_child_sessions(cid, age_group, profile)
        all_sessions.extend(sessions)
        all_children.append(meta)

    sessions_df = pd.DataFrame(all_sessions)
    children_df = pd.DataFrame(all_children)

    return sessions_df, children_df


if __name__ == "__main__":
    sessions_df, children_df = generate_dataset(500)
    print(f"Generated {len(children_df)} children, {len(sessions_df)} sessions")
    print(f"\nProfile distribution:\n{children_df['profile'].value_counts()}")
    print(f"\nAge group distribution:\n{children_df['age_group'].value_counts()}")
    print(f"\nLabel distribution:\n{children_df['label'].value_counts()}")
    print(f"\nSample session:\n{sessions_df.iloc[0]}")
