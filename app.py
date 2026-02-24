"""
Flask Web Application for CECI Testing UI.
Interactive cognitive games with telemetry capture and CECI dashboard.
"""

import os
import sys
import json
import numpy as np
from flask import Flask, render_template, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import CECIPredictor
from data.synthetic_generator import AGE_GROUP_GAMES

app = Flask(__name__)

# Load trained models
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        predictor = CECIPredictor.load(MODELS_DIR)
    return predictor


@app.route("/")
def index():
    """Serve the main testing UI."""
    return render_template("index.html")


@app.route("/api/games/<age_group>")
def get_games(age_group):
    """Get available games for an age group."""
    p = get_predictor()
    games = p.get_available_games(age_group)
    return jsonify(games)


@app.route("/api/games")
def get_all_games():
    """Get all available games across age groups."""
    p = get_predictor()
    games = p.get_available_games()
    return jsonify(games)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Run CECI prediction on submitted session data.
    Expects JSON: {
        "sessions": [{ accuracy, mean_reaction_time, hesitation_ratio,
                       task_completion_rate, error_burst_rate, engagement_score }, ...],
        "child_id": 1,
        "age_group": "3-5"
    }
    """
    data = request.get_json()
    if not data or "sessions" not in data:
        return jsonify({"error": "Missing sessions data"}), 400

    p = get_predictor()
    result = p.predict_from_sessions(
        sessions=data["sessions"],
        child_id=data.get("child_id", 1),
        age_group=data.get("age_group", "3-5"),
    )
    return jsonify(result)


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """
    Generate a simulated child and predict CECI.
    Expects JSON: { "age_group": "3-5", "profile": "typical", "n_sessions": 7 }
    """
    data = request.get_json() or {}
    p = get_predictor()
    result = p.predict_demo(
        age_group=data.get("age_group", "3-5"),
        profile=data.get("profile", "typical"),
        n_sessions=data.get("n_sessions", 7),
    )
    return jsonify(result)


if __name__ == "__main__":
    print("Loading CECI models...")
    get_predictor()
    print("Starting CECI Testing UI at http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
