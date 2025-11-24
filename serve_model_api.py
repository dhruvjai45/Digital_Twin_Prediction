#!/usr/bin/env python3
import os
import argparse
import json
import traceback
from typing import List, Dict, Any, Optional

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Model API")

# Allow local requests (change origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for localhost demo. lock this down in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = None
MODEL_OBJ = None
PREPROCESSOR = None
CLF = None
META = None  # loaded from best_pipeline.joblib.meta.json if present
EXPECTED_FEATURES = None  # list of features to present to preprocessor

class RecordBatch(BaseModel):
    # Accept either a single record or list - we will normalize
    records: List[Dict[str, Any]]

def load_meta(joblib_path: str) -> Optional[dict]:
    meta_path = joblib_path + ".meta.json"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def load_model(path: str):
    global MODEL_OBJ, PREPROCESSOR, CLF, META, EXPECTED_FEATURES
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    MODEL_OBJ = joblib.load(path)
    META = load_meta(path) or {}
    # if saved as dict {'pre':..., 'clf':...}
    if isinstance(MODEL_OBJ, dict):
        PREPROCESSOR = MODEL_OBJ.get('pre')
        CLF = MODEL_OBJ.get('clf')
    else:
        # if pipeline or estimator
        PREPROCESSOR = None
        CLF = MODEL_OBJ

    # If meta lists numeric & categorical features, build expected columns list
    if META:
        num = META.get("numeric_features", []) or []
        cat = META.get("categorical_features", []) or []
        EXPECTED_FEATURES = num + cat
    else:
        EXPECTED_FEATURES = None

    print("Model loaded. Preprocessor:", type(PREPROCESSOR).__name__ if PREPROCESSOR else None,
          "Classifier:", type(CLF).__name__)

def ensure_input_frame(records: List[Dict[str, Any]]) -> pd.DataFrame:
    # Build a DataFrame with expected features if meta exists; else use records as-is
    df = pd.DataFrame(records)
    if EXPECTED_FEATURES:
        # create frame with those columns (fill missing with NaN)
        out = pd.DataFrame({c: df.get(c, np.nan) for c in EXPECTED_FEATURES})
        # keep any extra columns too (optional)
        extra = [c for c in df.columns if c not in out.columns]
        if extra:
            out = pd.concat([out, df[extra]], axis=1)
        return out
    else:
        # Nothing known about expected features; return as-is
        return df

def safe_transform_and_predict(df: pd.DataFrame, threshold: float = 0.5):
    if PREPROCESSOR is None:
        # model may be a full pipeline that accepts raw DF
        try:
            proba = CLF.predict_proba(df)[:,1]
        except Exception as e:
            # last resort: try to convert DataFrame to numpy if classifier expects arrays
            try:
                proba = CLF.predict_proba(df.values)[:,1]
            except Exception as e2:
                raise RuntimeError(f"Model predict_proba failed: {e} || {e2}")
    else:
        # Preprocessor exists
        try:
            Xt = PREPROCESSOR.transform(df)
        except Exception as e:
            # helpful error with traceback
            tb = traceback.format_exc()
            raise RuntimeError(f"Preprocessor transform failed: {e}\n{tb}")
        try:
            proba = CLF.predict_proba(Xt)[:,1]
        except Exception as e:
            # try classifier on arrays
            try:
                proba = CLF.predict_proba(Xt)[:,1]
            except Exception as e2:
                raise RuntimeError(f"Classifier predict_proba failed: {e} || {e2}")
    preds = (proba >= threshold).astype(int)
    return preds.tolist(), proba.tolist()

@app.on_event("startup")
def startup_event():
    global MODEL_PATH
    if MODEL_PATH is None:
        return
    load_model(MODEL_PATH)

@app.get("/", tags=["status"])
def root():
    info = {
        "status": "ok",
        "model_path": MODEL_PATH,
        "has_preprocessor": PREPROCESSOR is not None,
        "has_classifier": CLF is not None,
        "meta_loaded": bool(META)
    }
    return info

@app.post("/predict", tags=["predict"])
def predict(payload: RecordBatch = Body(...), threshold: float = 0.5):
    """
    Predict endpoint.
    Body: { "records": [ {feature1: value1, feature2: value2, ...}, {...} ] }
    Query param: threshold (default 0.5)
    Response: {"results": [{"pred":0,"prob":0.12}, ...]}
    """
    try:
        records = payload.records
        if not isinstance(records, list) or len(records) == 0:
            raise HTTPException(status_code=400, detail="`records` must be a non-empty list of objects.")
        df = ensure_input_frame(records)
        preds, probs = safe_transform_and_predict(df, threshold=threshold)
        results = [{"pred": int(p), "prob": float(prob)} for p, prob in zip(preds, probs)]
        return {"results": results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional health route that returns metadata if .meta.json present
@app.get("/model_meta", tags=["status"])
def model_meta():
    if META:
        return META
    else:
        return {"meta": None}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="./best_pipeline.joblib", help="Path to joblib model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    MODEL_PATH = args.model
    print("Starting server, loading model:", MODEL_PATH)
    load_model(MODEL_PATH)
    import uvicorn
    uvicorn.run("serve_model_api:app", host=args.host, port=args.port, log_level="info")