import os
import joblib
import numpy as np

# ===============================
# Constants (must match training)
# ===============================
INTENTS = ["documents", "eligibility", "process", "timeline", "fees", "correction"]
SERVICES = [
    "ration_card",
    "birth_certificate",
    "unemployment_allowance"
]

# ===============================
# Absolute path resolution
# ===============================
# This file is: backend/next_step_recommender.py
# Project root is one level ABOVE backend/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "next_step_recommender",
    "next_step_model.pkl"
)

LABEL_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "next_step_recommender",
    "label_binarizer.pkl"
)

# Debug prints (keep for now)
print("🔹 Loading Next-Step Model from:", MODEL_PATH)
print("🔹 Loading Label Binarizer from:", LABEL_PATH)

# ===============================
# Load trained artifacts ONCE
# ===============================
model = joblib.load(MODEL_PATH)
mlb = joblib.load(LABEL_PATH)

# ===============================
# Inference function
# ===============================
def recommend_next_steps(service: str, current_intent: str, top_k: int = 3):
    """
    Recommend next steps (intents) based on current service and intent.
    """

    if service not in SERVICES or current_intent not in INTENTS:
        return []

    service_vec = [1 if service == s else 0 for s in SERVICES]
    intent_vec = [1 if current_intent == i else 0 for i in INTENTS]

    X = np.array([service_vec + intent_vec])
    probs = model.predict_proba(X)[0]

    ranked = sorted(
        zip(INTENTS, probs),
        key=lambda x: x[1],
        reverse=True
    )

    return [intent for intent, _ in ranked[:top_k]]
