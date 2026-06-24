from fastapi import FastAPI
import joblib

app=FastAPI()

model=joblib.load("models/credit_risk_model.pkl")
scaler=joblib.load("models/scaler.pkl")

@app.get("/")
def home():
    return {"message":"Credit Risk API Running"}