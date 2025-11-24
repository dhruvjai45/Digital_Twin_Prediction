from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import uuid

MODEL_PATH = "best_pipeline.joblib"
model_bundle = joblib.load(MODEL_PATH)

preprocessor = model_bundle["pre"]
clf = model_bundle["clf"]

app = FastAPI()

# ------------ INPUT MODEL ------------
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
    patient_id: int | None = None   # optional

# ------------ INTERNAL FEATURE BUILDER ------------
def do_feature_eng(df):
    df["pulse_pressure"] = df["blood_pressure_systolic"] - df["blood_pressure_diastolic"]
    df["sys_dia_ratio"] = df["blood_pressure_systolic"] / (df["blood_pressure_diastolic"] + 0.01)
    df["hr_spo2"] = df["heart_rate"] * df["oxygen_saturation"]
    df["age_sq"] = df["age"] ** 2
    df["age_bin"] = pd.cut(df["age"], bins=[0,25,40,55,70,200], labels=False).astype(float)
    df["missing_count"] = df.isna().sum(axis=1)
    return df

# ------------ API ROUTE ------------
@app.post("/predict")
def predict(data: PatientData):

    row = data.dict()
    patient_id = row.pop("patient_id", None)

    df = pd.DataFrame([row])
    df = do_feature_eng(df)

    X = preprocessor.transform(df)
    prob = clf.predict_proba(X)[0][1]

    # ---- determine descriptive fields ----
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
        "description": description,
        "level": level,
        "name": "Readmission Risk Predictor",
        "updated_at": datetime.utcnow().isoformat(),
        "value": round(float(prob), 4),
        "patient_id": patient_id
    }
