from pydantic import BaseModel

class CreditApplication(BaseModel):
    income: float
    credit_amount: float
    age: int
    previous_defaults: int

class CreditResponse(BaseModel):
    default_probability: float
    risk_level: str
    decision: str
