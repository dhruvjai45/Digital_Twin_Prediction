# server_model_api.py
"""
FastAPI wrapper for model serving.
Expose endpoint POST /predict
Request body:
{
  "records": [ {...}, {...} ]
}
Response:
{
  "predictions": [
    {
      "risk_score": 0.83,
      "prediction": 1,
      "alert_level": "High",
      "description": "High risk of abnormality",
      "name": "Digital Twin Health Risk",
      "patient_id": null,
      "updated_at": "2025-11-24T18:40:12"
    }
  ]
}
"""
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback

from server_model import PredictModel

# ------------- Configuration -------------
MODEL_PATH = os.environ.get("MODEL_PATH", "best_pipeline.joblib")
META_PATH = os.environ.get("MODEL_META_PATH", MODEL_PATH + ".meta.json")
# -----------------------------------------

app = FastAPI(title="Digital Twin Prediction API", version="1.0")

# Allow all origins for testing; in production lock this down.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic request model
class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]

# Initialize model once on startup
predictor: Optional[PredictModel] = None

@app.on_event("startup")
def startup_event():
    global predictor
    try:
        predictor = PredictModel(model_path=MODEL_PATH, meta_path=META_PATH)
        print("Model loaded:", MODEL_PATH)
    except Exception as e:
        # keep predictor as None; endpoints will raise informative error
        predictor = None
        print("Failed to load model at startup:", e)

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Accepts JSON body with "records": [ {...}, ... ]
    Returns structured predictions list.
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")
    try:
        records = req.records
        if not isinstance(records, list) or len(records) == 0:
            raise HTTPException(status_code=400, detail="records must be a non-empty list.")
        # get probabilities
        try:
            probs = predictor.predict_proba_for_records(records)
        except Exception as e:
            tb = traceback.format_exc()
            raise HTTPException(status_code=500, detail=f"Model predict_proba failed: {e}\n{tb}")

        results = []
        # patient_id extraction safe
        for rec, p in zip(records, probs):
            pid = rec.get("patient_id") if isinstance(rec, dict) else None
            out = predictor.prepare_prediction_record(prob=p, patient_id=pid)
            results.append(out)
        return {"predictions": results}
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}\n{tb}")

@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_PATH}
