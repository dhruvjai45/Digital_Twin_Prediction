# --------------------------------------------------------
# app.py (Final Clean Version – NO ANOMALY MODEL)
# For model: best_pipeline.joblib
# --------------------------------------------------------

import os
import json
from datetime import datetime
from typing import Optional, Union, Dict, Any
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Optional Groq AI
try:
    from groq import Groq
    HAS_GROQ = True
except:
    HAS_GROQ = False

load_dotenv()

# --------------------------------------------------------
# Groq Config
# --------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

groq_client = None
if HAS_GROQ and GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

# --------------------------------------------------------
# MODEL FILES
# --------------------------------------------------------
MODEL_LOCAL = "best_pipeline.joblib"

print("\nLoading best_pipeline.joblib...")

try:
    loaded = joblib.load(MODEL_LOCAL)
    print("Loaded:", type(loaded))
except Exception as e:
    raise RuntimeError(f"❌ Could not load best_pipeline.joblib: {e}")

# --------------------------------------------------------
# Extract model components
# --------------------------------------------------------
if not isinstance(loaded, dict):
    raise RuntimeError("❌ best_pipeline.joblib must contain a dict with keys: pre, regressor")

preprocessor = loaded.get("pre", None)
model = loaded.get("regressor", None)

if preprocessor is None:
    raise RuntimeError("❌ Missing 'pre' in best_pipeline.joblib")

if model is None:
    raise RuntimeError("❌ Missing 'regressor' in best_pipeline.joblib")

# Feature names
FEATURE_NAMES = getattr(preprocessor, "feature_names_in_", None)
if FEATURE_NAMES is not None:
    FEATURE_NAMES = list(FEATURE_NAMES)


# --------------------------------------------------------
# FastAPI initialization
# --------------------------------------------------------
app = FastAPI(
    title="Digital Twin Prediction API",
    version="2.0",
)

# serve UI if exists
if Path("index.html").exists():
    app.mount("/static", StaticFiles(directory="."), name="static")
    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse("index.html")


# --------------------------------------------------------
# Request schema
# --------------------------------------------------------
class PatientInput(BaseModel):
    patient_id: Union[str, int]
    age: float
    weight: float
    blood_pressure: str
    heart_rate: float
    respiration: float
    spo2: float
    temperature: float
    glucose: float
    comorbidity_count: float
    prior_admissions_90d: float
    timestamp: Optional[str] = None
    device_id: Optional[str] = None


# --------------------------------------------------------
# Helper Functions
# --------------------------------------------------------
def parse_blood_pressure(bp_str: str):
    try:
        s, d = bp_str.replace(" ", "").split("/")
        return float(s), float(d)
    except:
        raise HTTPException(status_code=400, detail="Invalid blood_pressure format. Use '120/80'.")

def build_feature_dataframe(data: PatientInput) -> pd.DataFrame:
    sbp, dbp = parse_blood_pressure(data.blood_pressure)

    row = {
        "age": data.age,
        "weight": data.weight,
        "blood_pressure_systolic": sbp,
        "blood_pressure_diastolic": dbp,
        "blood_pressure": data.blood_pressure,
        "heart_rate": data.heart_rate,
        "respiratory_rate": data.respiration,
        "oxygen_saturation": data.spo2,
        "temperature": data.temperature,
        "glucose_fasting": data.glucose,
        "glucose": data.glucose,
        "comorbidity_count": data.comorbidity_count,
        "prior_admissions_90d": data.prior_admissions_90d,
        "device_id": data.device_id or "unknown",
        "missing_dummy": 0.0,
    }

    df = pd.DataFrame([row])

    # Derived features
    df["pulse_pressure"] = df["blood_pressure_systolic"] - df["blood_pressure_diastolic"]

    # FIXED (correct sys/dia ratio)
    df["sys_dia_ratio"] = df["blood_pressure_systolic"] / df["blood_pressure_diastolic"].replace(0, np.nan)

    df["hr_spo2"] = df["heart_rate"] * df["oxygen_saturation"]
    df["age_sq"] = df["age"] ** 2
    df["age_bin"] = pd.cut(df["age"], bins=[0,25,40,55,70,200], labels=False)
    df["missing_count"] = df.isna().sum(axis=1)

    return df

def ai_generate_description(name: str, value: Any) -> str:
    if groq_client is None:
        return f"{name} = {value}."

    prompt = f"""
Give a short clinical interpretation for:
{name}: {value}
"""

    try:
        out = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.2,
        )
        return out.choices[0].message.content.strip()
    except:
        return f"{name} = {value}."


def ai_generate_summary(payload: Dict[str, Any]):
    if groq_client is None:
        return "AI summary unavailable."

    prompt = f"""
Summarize this patient's vitals and risk:

{json.dumps(payload)}

Write 4–6 clear medical sentences.
"""

    try:
        out = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role":"user","content":prompt}],
            max_tokens=200,
            temperature=0.25,
        )
        return out.choices[0].message.content.strip()
    except:
        return "Summary unavailable."


def risk_level(prob: float) -> str:
    if prob < 0.33: return "low"
    if prob < 0.66: return "medium"
    return "high"


def level_for_observation(name: str, v: float) -> str:
    if v is None: return "unknown"

    try:
        v = float(v)
    except:
        return "unknown"

    if name == "heart_rate":
        if v < 30: return "critical"
        if v < 50: return "low"
        if v <= 100: return "normal"
        if v <= 140: return "high"
        return "critical"

    if name == "oxygen_saturation":
        if v >= 95: return "normal"
        if v >= 90: return "low"
        return "critical"

    if name == "respiratory_rate":
        if v < 8: return "critical"
        if v < 12: return "low"
        if v <= 20: return "normal"
        if v <= 30: return "high"
        return "critical"

    if name == "temperature":
        if v < 95: return "critical"
        if v <= 99.5: return "normal"
        if v <= 102: return "high"
        return "critical"

    if name in ("glucose", "glucose_fasting"):
        if v < 70: return "low"
        if v < 100: return "normal"
        if v < 126: return "elevated"
        return "high"

    if name == "blood_pressure_systolic":
        if v >= 180: return "critical"
        if v >= 140: return "high"
        if v >= 130: return "elevated"
        if v < 90: return "low"
        return "normal"

    if name == "blood_pressure_diastolic":
        if v >= 120: return "critical"
        if v >= 90: return "high"
        if v >= 80: return "elevated"
        if v < 60: return "low"
        return "normal"

    if name == "weight":
        if v < 30: return "low"
        if v > 200: return "high"
        return "normal"

    if name == "readmission_risk":
        return risk_level(v)

    return "unknown"


# --------------------------------------------------------
# Prediction Route
# --------------------------------------------------------
@app.post("/predict")
def predict(data: PatientInput):

    # Build features
    df = build_feature_dataframe(data)

    # Align with preprocessor
    if FEATURE_NAMES:
        for col in FEATURE_NAMES:
            if col not in df.columns:
                df[col] = np.nan
        df = df[FEATURE_NAMES]

    # Apply preprocessing
    try:
        X = preprocessor.transform(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessor failed: {e}")

    # Predict (LGBMRegressor gives .predict(), NOT predict_proba)
    try:
        proba = float(model.predict(X)[0])  # READMISSION PROBABILITY
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    label = int(proba >= 0.5)
    risk = risk_level(proba)

    # Build observations
    sbp, dbp = parse_blood_pressure(data.blood_pressure)
    vitals = {
        "blood_pressure_systolic": sbp,
        "blood_pressure_diastolic": dbp,
        "heart_rate": data.heart_rate,
        "oxygen_saturation": data.spo2,
        "respiratory_rate": data.respiration,
        "temperature": data.temperature,
        "glucose_fasting": data.glucose,
        "weight": data.weight,
        "readmission_risk": proba,
    }

    time_now = datetime.utcnow().isoformat() + "Z"

    observations = []
    for name, value in vitals.items():
        observations.append({
            "name": name,
            "value": value,
            "level": level_for_observation(name, value),
            "description": ai_generate_description(name, value),
            "updated_at": time_now,
            "patient_id": str(data.patient_id),
        })

    # Summary
    result = {
        "status": "ok",
        "patient_id": str(data.patient_id),
        "probability": proba,
        "predicted_label": label,
        "risk": risk,
        "observations": observations,
        "override_applied": False,
        "description": ai_generate_summary({
            "probability": proba,
            "risk": risk,
            "vitals": vitals
        }),
        "timestamp": time_now
    }

    return result


# --------------------------------------------------------
# Health check
# --------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}
