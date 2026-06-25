from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app=FastAPI()

class CreditData(BaseModel):
    checking_account: str
    duration: int
    credit_history: str
    purpose: str
    credit_amount: float
    savings: str
    employment: str
    installment_rate: int
    personal_status_sex: str
    other_debtors: str
    residence_since: int
    property: str
    age: int
    other_installment_plans: str
    housing: str
    existing_credits: int
    job: str
    dependents: int
    telephone: str
    foreign_worker: str


model=joblib.load("models/credit_risk_model.pkl")
scaler=joblib.load("models/scaler.pkl")
feature_names=joblib.load("models/feature_names.pkl")

@app.get("/")
def home():
    return {"message":"Credit Risk API Running"}

@app.post("/predict")
def predict(data:CreditData):
    input_df=pd.DataFrame([data.model_dump()])

    input_df=pd.get_dummies(input_df)

    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)

    if(prediction[0]==1):
        result="Good Credit"
    else:
        result="Bad Credit"
    return {
        "prediction":result,
        "confidence":probability
    }