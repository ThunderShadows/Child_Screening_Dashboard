"""
CECI Index Computation Module.
Implements the Child Effort-Cognition Index (Equation 4).
"""

import numpy as np
from typing import Dict, Tuple, List


# Default weights (normalized, sum to 1.0)
DEFAULT_WEIGHTS = {
    "w1": 0.50,  # PID weight (persistent cognitive difficulty)
    "w2": 0.30,  # Consistency weight (1 - Var(Acc))
    "w3": 0.20,  # PEff weight (low effort penalty reduction)
}

# Risk band thresholds
RISK_BANDS = {
    "green": {"min": 0.0, "max": 0.40, "label": "Low Risk", "color": "#22c55e"},
    "amber": {"min": 0.40, "max": 0.65, "label": "Moderate Risk - Monitor", "color": "#f59e0b"},
    "red":   {"min": 0.65, "max": 1.0, "label": "High Risk - Refer", "color": "#ef4444"},
}


def compute_ceci(
    pid: float,
    var_acc: float,
    peff: float,
    weights: Dict[str, float] = None,
) -> float:
    """
    Compute the Child Effort-Cognition Index (Equation 4).

    CECI = w1 * PID + w2 * (1 - Var(Acc)) - w3 * PEff

    Args:
        pid: Probability of persistent cognitive difficulty [0, 1]
        var_acc: Cross-session accuracy variance (Equation 3) [0, 1]
        peff: Probability of low/inconsistent effort [0, 1]
        weights: Custom weights dict {w1, w2, w3}

    Returns:
        CECI score clipped to [0, 1]
    """
    w = weights or DEFAULT_WEIGHTS
    ceci = w["w1"] * pid + w["w2"] * (1.0 - var_acc) - w["w3"] * peff
    return float(np.clip(ceci, 0.0, 1.0))


def compute_ceci_batch(
    pid_array: np.ndarray,
    var_acc_array: np.ndarray,
    peff_array: np.ndarray,
    weights: Dict[str, float] = None,
) -> np.ndarray:
    """Vectorized CECI computation for a batch of children."""
    w = weights or DEFAULT_WEIGHTS
    ceci = w["w1"] * pid_array + w["w2"] * (1.0 - var_acc_array) - w["w3"] * peff_array
    return np.clip(ceci, 0.0, 1.0)


def classify_risk_band(ceci: float) -> Dict:
    """
    Map CECI score to a risk band (green / amber / red).

    Returns:
        Dict with band name, label, color, and CECI score.
    """
    for band_name, band_info in RISK_BANDS.items():
        if band_info["min"] <= ceci < band_info["max"] or (
            band_name == "red" and ceci >= band_info["min"]
        ):
            return {
                "band": band_name,
                "label": band_info["label"],
                "color": band_info["color"],
                "ceci_score": round(ceci, 4),
            }
    # Fallback
    return {
        "band": "green",
        "label": RISK_BANDS["green"]["label"],
        "color": RISK_BANDS["green"]["color"],
        "ceci_score": round(ceci, 4),
    }


def classify_risk_band_batch(ceci_array: np.ndarray) -> List[Dict]:
    """Classify risk bands for a batch of CECI scores."""
    return [classify_risk_band(c) for c in ceci_array]


def get_ceci_breakdown(
    pid: float,
    var_acc: float,
    peff: float,
    uncertainty: float = 0.0,
    weights: Dict[str, float] = None,
) -> Dict:
    """
    Get a full interpretable breakdown of the CECI score.

    Returns:
        Complete breakdown dict suitable for clinical dashboards.
    """
    w = weights or DEFAULT_WEIGHTS
    ceci = compute_ceci(pid, var_acc, peff, weights)
    risk = classify_risk_band(ceci)

    # Component contributions
    pid_contribution = w["w1"] * pid
    consistency_contribution = w["w2"] * (1.0 - var_acc)
    effort_adjustment = w["w3"] * peff

    return {
        "ceci_score": round(ceci, 4),
        "risk_band": risk["band"],
        "risk_label": risk["label"],
        "risk_color": risk["color"],
        "components": {
            "pid": {
                "value": round(pid, 4),
                "weight": w["w1"],
                "contribution": round(pid_contribution, 4),
                "interpretation": _interpret_pid(pid),
            },
            "consistency": {
                "value": round(1.0 - var_acc, 4),
                "variance": round(var_acc, 4),
                "weight": w["w2"],
                "contribution": round(consistency_contribution, 4),
                "interpretation": _interpret_consistency(var_acc),
            },
            "effort": {
                "value": round(peff, 4),
                "weight": w["w3"],
                "adjustment": round(effort_adjustment, 4),
                "interpretation": _interpret_effort(peff),
            },
        },
        "uncertainty": round(uncertainty, 4),
        "confidence": round(1.0 - uncertainty, 4),
        "clinical_note": _generate_clinical_note(ceci, pid, var_acc, peff, risk["band"]),
    }


def _interpret_pid(pid: float) -> str:
    if pid > 0.7:
        return "Strong indicators of persistent cognitive difficulty across sessions"
    elif pid > 0.4:
        return "Moderate indicators of cognitive difficulty; further monitoring recommended"
    else:
        return "Low indicators of persistent cognitive difficulty"


def _interpret_consistency(var_acc: float) -> str:
    if var_acc > 0.05:
        return "High session-to-session variability suggests inconsistent performance"
    elif var_acc > 0.02:
        return "Moderate consistency across sessions"
    else:
        return "Highly consistent performance across sessions"


def _interpret_effort(peff: float) -> str:
    if peff > 0.6:
        return "Strong signs of inconsistent effort or engagement fluctuation"
    elif peff > 0.3:
        return "Some variability in effort and engagement detected"
    else:
        return "Effort and engagement appear consistent"


def _generate_clinical_note(
    ceci: float, pid: float, var_acc: float, peff: float, band: str,
) -> str:
    if band == "red":
        return (
            f"CECI score of {ceci:.2f} indicates HIGH RISK. "
            f"Persistent cognitive difficulty probability ({pid:.0%}) is elevated "
            f"with {'stable' if var_acc < 0.03 else 'variable'} performance patterns. "
            f"Referral for comprehensive assessment is recommended."
        )
    elif band == "amber":
        return (
            f"CECI score of {ceci:.2f} indicates MODERATE RISK. "
            f"Continued monitoring through additional game sessions is advised. "
            f"{'Effort variability detected — poor performance may partly reflect engagement issues.' if peff > 0.4 else 'Performance patterns warrant further observation.'}"
        )
    else:
        return (
            f"CECI score of {ceci:.2f} indicates LOW RISK. "
            f"Performance is within expected range. "
            f"{'Note: some effort fluctuation detected.' if peff > 0.3 else 'Engagement appears consistent.'}"
        )
