# app.py (updated: observations now include 'level' for each item)
import os
import json
from datetime import datetime
from typing import Optional, Union, List, Dict, Any
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

# Try import groq SDK; if not available we will still run (fallback descriptions)
try:
    from groq import Groq
    HAS_GROQ = True
except Exception:
    HAS_GROQ = False

load_dotenv()

# ----------------------------
# Config / env
# ----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Model files (LOCAL relative paths – correct for Render)
MODEL_LOCAL = "xgboost_model.joblib"
PRE_LOCAL = "preprocessor.joblib"
ANOMALY_LOCAL = "anomaly_model.joblib"

print("Anomaly file exists? ->", Path(ANOMALY_LOCAL).exists())


# Optional URLs to download from
MODEL_URL = os.getenv("MODEL_URL")
PREPROCESSOR_URL = os.getenv("PREPROCESSOR_URL")
ANOMALY_URL = os.getenv("ANOMALY_URL")

# ----------------------------
# Groq client (if key present)
# ----------------------------
groq_client = None
if HAS_GROQ and GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

# ----------------------------
# Helper: download models at runtime if URLs provided
# ----------------------------
def download_file_if_needed(url: Optional[str], dest_path: str) -> bool:
    dest = Path(dest_path)
    if dest.exists() and dest.stat().st_size > 0:
        return True
    if not url:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return True

# try downloads
try:
    if MODEL_URL:
        download_file_if_needed(MODEL_URL, MODEL_LOCAL)
    if PREPROCESSOR_URL:
        download_file_if_needed(PREPROCESSOR_URL, PRE_LOCAL)
    if ANOMALY_URL:
        download_file_if_needed(ANOMALY_URL, ANOMALY_LOCAL)
except Exception as e:
    # log but don't crash at this stage; load below will raise if missing
    print("Warning: download failure:", e)

# ----------------------------
# Load models (joblib)
# ----------------------------
try:
    model = joblib.load(MODEL_LOCAL)
except Exception as e:
    raise RuntimeError(f"Could not load main model from {MODEL_LOCAL}: {e}")

try:
    preprocessor = joblib.load(PRE_LOCAL)
except Exception as e:
    raise RuntimeError(f"Could not load preprocessor from {PRE_LOCAL}: {e}")

# anomaly model is optional
anomaly_model = None
if Path(ANOMALY_LOCAL).exists():
    try:
        anomaly_model = joblib.load(ANOMALY_LOCAL)
    except Exception as e:
        print("Warning: could not load anomaly model:", e)

# columns preprocessor expects (safe handling)
FEATURE_NAMES = getattr(preprocessor, "feature_names_in_", None)
if FEATURE_NAMES is not None:
    FEATURE_NAMES = list(FEATURE_NAMES)

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="Digital Twin Prediction API (Groq Enhanced)", version="1.0")

# serve index.html statically (if present)
if Path("index.html").exists():
    app.mount("/static", StaticFiles(directory="."), name="static")
    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse("index.html")

# ----------------------------
# Request schema
# ----------------------------
class PatientInput(BaseModel):
    patient_id: Union[str, int]
    age: float
    weight: float
    blood_pressure: str  # e.g. "118/76"
    heart_rate: float
    respiration: float
    spo2: float
    temperature: float
    glucose: float
    comorbidity_count: float
    prior_admissions_90d: float
    timestamp: Optional[str] = None
    device_id: Optional[str] = None

# ----------------------------
# Helpers
# ----------------------------
def parse_blood_pressure(bp_str: str):
    try:
        s, d = bp_str.replace(" ", "").split("/")
        return float(s), float(d)
    except Exception:
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

    # derived features (match training)
    try:
        df["pulse_pressure"] = df["blood_pressure_systolic"] - df["blood_pressure_diastolic"]
    except Exception:
        df["pulse_pressure"] = np.nan
    try:
        df["sys_dia_ratio"] = df["blood_pressure_systolic"] / df["blood_pressure_diastolic"].replace(0, np.nan)
    except Exception:
        df["sys_dia_ratio"] = np.nan
    try:
        df["hr_spo2"] = df["heart_rate"] * df["oxygen_saturation"]
    except Exception:
        df["hr_spo2"] = np.nan
    try:
        df["age_sq"] = df["age"] ** 2
    except Exception:
        df["age_sq"] = np.nan
    try:
        df["age_bin"] = pd.cut(df["age"], bins=[0,25,40,55,70,200], labels=False)
    except Exception:
        df["age_bin"] = 2
    df["missing_count"] = df.isna().sum(axis=1)
    return df

def ai_generate_description(observation_name: str, value: Any) -> str:
    """
    Generates a concise description using Groq if available,
    otherwise returns a safe fallback.
    """
    prompt = f"""
You are a clinical assistant.

Observation: {observation_name}
Value: {value}

In 1-2 sentences, say whether this value is normal, borderline, or concerning and what it commonly indicates clinically.
Be concise, professional.
"""
    if groq_client is None:
        return f"No AI description available. Observation {observation_name} = {value}."
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role":"user","content":prompt}],
            max_tokens=120,
            temperature=0.2,
        )
        text = completion.choices[0].message.content
        return text.strip()
    except Exception as e:
        return f"AI description unavailable (error: {e}). Observation {observation_name} = {value}."

def ai_generate_summary(result_payload: Dict[str, Any]) -> str:
    prompt = f"""
You are a medical decision-support assistant.

Summarize the following JSON about a patient's vitals and model readmission risk:

{json.dumps(result_payload, ensure_ascii=False)}

Write 4-6 sentences:
- comment on vital stability/abnormalities
- interpret readmission risk and probability
- remind that this is decision support, not a diagnosis
"""
    if groq_client is None:
        return "AI summary not available. Interpret results using the observations and probability."
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role":"user","content":prompt}],
            max_tokens=220,
            temperature=0.25,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Summary generation failed (error: {e}). Use raw values to interpret the case."

def risk_level(prob: float) -> str:
    if prob < 0.33:
        return "low"
    if prob < 0.66:
        return "medium"
    return "high"

def level_for_observation(name: str, value: Any, readmission_risk_level: str = None, anomaly_flag: str = None) -> Optional[str]:
    """
    Return a human-readable level for an observation value.
    Levels: 'low'/'normal'/'elevated'/'high'/'critical'/'unknown'
    This function uses conservative, generic thresholds.
    """
    if value is None:
        return "unknown"

    # numeric coercion where possible
    try:
        v = float(value)
    except Exception:
        # non-numeric observations (e.g., device_id) -> unknown
        if name == "readmission_risk":
            return readmission_risk_level or "unknown"
        if name == "anomaly_score":
            return anomaly_flag or "unknown"
        return "unknown"

    # thresholds by observation
    if name == "blood_pressure_systolic":
        if v >= 180:
            return "critical"
        if v >= 140:
            return "high"
        if v >= 130:
            return "elevated"
        if v >= 120:
            return "elevated"
        if v < 90:
            return "low"
        return "normal"

    if name == "blood_pressure_diastolic":
        if v >= 120:
            return "critical"
        if v >= 90:
            return "high"
        if v >= 80:
            return "elevated"
        if v < 60:
            return "low"
        return "normal"

    if name == "heart_rate":
        if v < 50:
            return "low"
        if v <= 100:
            return "normal"
        if v <= 140:
            return "high"
        return "critical"

    if name == "oxygen_saturation":
        # SpO2 as percentage
        if v >= 95:
            return "normal"
        if 90 <= v < 95:
            return "low"
        return "critical"

    if name == "respiratory_rate":
        if v < 12:
            return "low"
        if 12 <= v <= 20:
            return "normal"
        if 21 <= v <= 30:
            return "high"
        return "critical"

    if name in ("temperature", "body_temperature", "temp"):
        # assuming Fahrenheit as your example values used F
        if v < 95:
            return "critical"
        if v <= 99.5:
            return "normal"
        if v <= 102:
            return "high"
        return "critical"

    if name in ("glucose_fasting", "glucose"):
        if v < 70:
            return "low"
        if v < 100:
            return "normal"
        if v < 126:
            return "elevated"
        return "high"

    if name == "weight":
        # generic fallback: extreme values only
        if v < 30:
            return "low"
        if v > 200:
            return "high"
        return "normal"

    if name == "readmission_risk":
        return readmission_risk_level or risk_level(v)

    if name == "anomaly_score":
        # treat negative decision function as anomaly (model-specific)
        if anomaly_flag:
            return anomaly_flag
        # fallback thresholds (model dependent) - more negative -> more anomalous
        try:
            if v < -3.0:
                return "critical"
            if v < -1.0:
                return "high"
            return "normal"
        except Exception:
            return "unknown"

    # default
    return "unknown"

# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {"status":"ok","timestamp": datetime.utcnow().isoformat() + "Z"}

@app.post("/predict")
def predict(data: PatientInput):
    # build features
    df = build_feature_dataframe(data)

    # align with preprocessor's feature list
    if FEATURE_NAMES:
        for c in FEATURE_NAMES:
            if c not in df.columns:
                df[c] = np.nan
        df = df[list(FEATURE_NAMES)]

    # transform
    try:
        X = preprocessor.transform(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessor transform failed: {e}")

    # predict
    try:
        proba = float(model.predict_proba(X)[0][1])
        label = int(proba >= 0.5)
        risk = risk_level(proba)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # anomaly
    anomaly_score = None
    anomaly_flag = "unknown"
    if anomaly_model is not None:
        try:
            # decision_function exists on IsolationForest; some models return anomaly score differently
            if hasattr(anomaly_model, "decision_function"):
                score = float(anomaly_model.decision_function(X)[0])
                anomaly_score = score
                anomaly_flag = "yes" if score < -0.3 else "no"
            elif hasattr(anomaly_model, "score_samples"):
                score = float(anomaly_model.score_samples(X)[0])
                anomaly_score = score
                anomaly_flag = "yes" if score < -3.0 else "no"
        except Exception as e:
            anomaly_score = None
            anomaly_flag = "unknown"

    # build observations
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
        "anomaly_score": anomaly_score,
    }

    now_iso = datetime.utcnow().isoformat() + "Z"
    observations = []
    for name, value in vitals.items():
        desc = ai_generate_description(name, value)
        lvl = level_for_observation(name, value, readmission_risk_level=risk, anomaly_flag=anomaly_flag)
        observations.append({
            "name": name,
            "value": value,
            "level": lvl,
            "description": desc,
            "updated_at": now_iso,
            "patient_id": str(data.patient_id),
        })

    result = {
        "status": "ok",
        "patient_id": str(data.patient_id),
        "probability": proba,
        "predicted_label": label,
        "risk": risk,
        "anomaly_score": anomaly_score,
        "anomaly_flag": anomaly_flag,
        "observations": observations
    }

    result["description"] = ai_generate_summary(result)
    result["timestamp"] = now_iso
    return result
