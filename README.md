# CECI Screening Model

**Child Effort-Cognition Index** — A behavioral game-based screening framework for early identification of intellectual disability in children aged 0–9 years.

## Overview

This system uses a **hybrid AI/ML pipeline** combining interpretable tree-based models with temporal deep learning to capture fine-grained behavioral telemetry from interactive cognitive micro-games. Unlike traditional one-time assessments, it models **temporal stability and variability** across multiple sessions to distinguish genuine cognitive impairment from low performance caused by fluctuating effort, attention, or situational factors.

### Key Innovation: CECI Index

```
CECI = w1 · PID + w2 · (1 − Var(Acc)) − w3 · PEff
```

| Component | Description |
|-----------|-------------|
| **PID** | Probability of persistent cognitive difficulty (from LSTM) |
| **Var(Acc)** | Cross-session accuracy variance — performance consistency |
| **PEff** | Probability of low/inconsistent effort |

The CECI score maps to risk bands: 🟢 **Green** (Low Risk) · 🟡 **Amber** (Monitor) · 🔴 **Red** (Refer for Assessment)

---

## Project Structure

```
├── data/
│   └── synthetic_generator.py      # Synthetic data with age-group-specific games
├── models/
│   ├── feature_engineering.py      # Session-level + longitudinal feature extraction
│   ├── tree_model.py               # XGBoost interpretable baseline classifier
│   ├── temporal_model.py           # LSTM dual-head model (PID + PEff)
│   ├── calibration.py              # Platt scaling probabilistic calibration
│   └── ceci_index.py               # CECI formula + risk band classification
├── training/
│   └── train_pipeline.py           # End-to-end training pipeline
├── inference.py                    # CECIPredictor API for integration
├── app.py                          # Flask web app for testing UI
├── templates/
│   └── index.html                  # Game-based testing interface
├── static/
│   ├── style.css                   # Premium dark-themed UI
│   └── app.js                      # Interactive game engine (12 games)
└── requirements.txt
```

---

## Cognitive Games by Age Group

| Age Group | Games | Cognitive Domains |
|-----------|-------|-------------------|
| **0–2 years** | Tap the Animal, Peek-a-Boo Memory, Shape Sorter | Auditory recognition, working memory, visual-spatial |
| **3–5 years** | Color Pattern Match, Memory Card Flip, Counting Garden, Story Sequence | Pattern recognition, working memory, numerical cognition, sequential reasoning |
| **6–9 years** | Digit Span Recall, Word Category Sort, Spatial Puzzle, Math Race, Go/No-Go | Working memory, verbal reasoning, visual-spatial, numerical cognition, inhibitory control |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python training/train_pipeline.py
```

This generates synthetic data (500 children), trains XGBoost + LSTM, calibrates probabilities, and saves model artifacts to `saved_models/`.

### 3. Run the Testing UI

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) to access the interactive game-based screening interface.

### 4. Use the Inference API (for integration)

```python
from inference import CECIPredictor

predictor = CECIPredictor.load("saved_models")

result = predictor.predict_from_sessions([
    {"accuracy": 0.4, "mean_reaction_time": 3.2, "hesitation_ratio": 0.15,
     "task_completion_rate": 0.4, "error_burst_rate": 0.3, "engagement_score": 0.5},
    # ... more sessions
], child_id=1, age_group="3-5")

print(result["ceci_score"])     # 0.72
print(result["risk_band"])      # "red"
print(result["clinical_note"])  # Full clinical interpretation
```

---

## ML Pipeline Architecture

```
Session Trials → Feature Engineering → ┬─ XGBoost (baseline risk)
                                        ├─ LSTM (PID + PEff)
                                        └─ Platt Calibration
                                              ↓
                                         CECI Index
                                              ↓
                                     Risk Band (Green/Amber/Red)
```

### Training Results

| Metric | Value |
|--------|-------|
| XGBoost CV AUC | 0.9998 ± 0.0003 |
| XGBoost CV Accuracy | 98.6% |
| LSTM Final Loss | 0.1881 |

---

## Disclaimer

This framework is a **decision-support tool**, not a diagnostic replacement. It is designed to assist early referral decisions and intervention planning, particularly in low-resource settings where specialist access is limited.

---

## License

MIT
