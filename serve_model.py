# serve_model.py

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import uuid
import os

# --------------------------
# Load Model
# --------------------------

MODEL_PATH = "D:\\D_desktop\\SWE_Project\\swe1\\Digital_Twin_Prediction\\best_pipeline.joblib"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")

bundle = joblib.load(MODEL_PATH)

preprocessor = bundle.get("pre", None)
clf = bundle.get("clf", None)

if clf is None:
    raise RuntimeError("Classifier missing inside joblib bundle!")
if preprocessor is None:
    raise RuntimeError("Preprocessor missing inside joblib bundle!")

# --------------------------
# FastAPI Application
# --------------------------

app = FastAPI(title="Digital Twin Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# Input Schema
# --------------------------

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


# --------------------------
# Extra Derived Features
# --------------------------

def add_features(df):
    df["pulse_pressure"] = df["blood_pressure_systolic"] - df["blood_pressure_diastolic"]
    df["sys_dia_ratio"] = df["blood_pressure_systolic"] / (df["blood_pressure_diastolic"] + 0.01)
    df["hr_spo2"] = df["heart_rate"] * df["oxygen_saturation"]
    df["age_sq"] = df["age"] ** 2
    df["age_bin"] = pd.cut(df["age"], bins=[0,25,40,55,70,200], labels=False).astype(float)
    df["missing_count"] = df.isna().sum(axis=1)
    return df


# --------------------------
# Prediction Route
# --------------------------

@app.post("/predict")
def predict(data: PatientData):

    df = pd.DataFrame([data.dict()])
    df = add_features(df)

    X = preprocessor.transform(df)
    prob = clf.predict_proba(X)[0][1]

    # Severity buckets
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
        "name": "Digital Twin Readmission Predictor",
        "value": round(float(prob), 4),
        "description": description,
        "level": level,
        "updated_at": datetime.utcnow().isoformat()
    }


@app.get("/")
def home():
    return {"status": "ok", "message": "Digital Twin Predictor API running!"}
