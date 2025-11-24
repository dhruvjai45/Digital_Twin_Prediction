from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import uuid
import os

# ---------------------------------------------------
# Load model on startup (Render-friendly)
# ---------------------------------------------------

MODEL_PATH = "best_pipeline.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model_bundle = joblib.load(MODEL_PATH)
preprocessor = model_bundle["pre"]
clf = model_bundle["clf"]

# ---------------------------------------------------
# FastAPI app
# ---------------------------------------------------

app = FastAPI(title="Digital Twin Prediction API")

class PatientData(BaseModel):
    age: float
    blood_pressure_systolic: float
    blood_pressure_diastolic: float
    heart_rate: float
    respiratory_rate: float
    body_temperature: float
    oxygen_saturation: float
    glucose_fasting: float
    cholesterol_total: float
    pain_score: float
    diabetes: int
    smoker: int
    gender: str
    patient_id: int | None = None

# ---------------------------------------------------
# Feature engineering (must match training)
# ---------------------------------------------------

def feature_engineering(df):
    df["pulse_pressure"] = df["blood_pressure_systolic"] - df["blood_pressure_diastolic"]
    df["sys_dia_ratio"] = df["blood_pressure_systolic"] / (df["blood_pressure_diastolic"] + 0.01)
    df["hr_spo2"] = df["heart_rate"] * df["oxygen_saturation"]
    df["age_sq"] = df["age"] ** 2
    df["age_bin"] = pd.cut(df["age"], bins=[0,25,40,55,70,200], labels=False).astype(float)
    df["missing_count"] = df.isna().sum(axis=1)
    return df

# ---------------------------------------------------
# Prediction route
# ---------------------------------------------------

@app.post("/predict")
def predict(data: PatientData):
    try:
        row = data.dict()
        patient_id = row.pop("patient_id", None)

        df = pd.DataFrame([row])
        df = feature_engineering(df)

        X = preprocessor.transform(df)
        prob = float(clf.predict_proba(X)[0][1])

        # Label levels
        if prob < 0.33:
            level = "low"
            description = "Low readmission risk"
        elif prob < 0.66:
            level = "medium"
            description = "Moderate readmission risk"
        else:
            level = "high"
            description = "High readmission risk"

        return {
            "id": str(uuid.uuid4()),
            "name": "Readmission Risk Predictor",
            "description": description,
            "value": round(prob, 4),
            "level": level,
            "updated_at": datetime.utcnow().isoformat(),
            "patient_id": patient_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------
# Root
# ---------------------------------------------------

@app.get("/")
def home():
    return {"status": "running", "model": "best_pipeline.joblib"}
