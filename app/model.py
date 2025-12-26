import joblib
import numpy as np

MODEL_PATH = "models/credit_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except:
    model = None

def predict_default(features: np.ndarray) -> float:
    if model:
        return model.predict_proba(features)[0][1]
    # fallback dummy
    return min(0.9, features.mean() / 100_000)

def business_decision(prob: float):
    if prob < 0.2:
        return "low", "approved"
    elif prob < 0.4:
        return "medium", "manual_review"
    else:
        return "high", "rejected"
