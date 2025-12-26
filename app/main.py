from fastapi import FastAPI
import numpy as np

from app.schemas import CreditApplication, CreditResponse
from app.model import predict_default, business_decision

app = FastAPI(
    title="Mini Plurall - Credit Scoring API",
    version="1.0"
)

@app.get("/")
def health():
    return {"status": "ok", "service": "mini-plurall"}

@app.post("/predict", response_model=CreditResponse)
def predict(application: CreditApplication):

    features = np.array([[
        application.income,
        application.credit_amount,
        application.age,
        application.previous_defaults
    ]])

    prob = predict_default(features)
    risk, decision = business_decision(prob)

    return CreditResponse(
        default_probability=round(prob, 3),
        risk_level=risk,
        decision=decision
    )
