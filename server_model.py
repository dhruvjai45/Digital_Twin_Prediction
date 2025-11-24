# server_model.py
"""
Model loader + prediction helper.
Designed to be robust to various saved bundle shapes:
 - FinalEnsemble-like object with predict_proba(X)
 - dict saved with keys: 'pre', 'clf' (preprocessor + sklearn clf)
 - sklearn Pipeline with predict_proba
Also reads accompanying .meta.json if present to get feature lists and threshold.
"""
import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

DEFAULT_THRESHOLD = 0.5

class PredictModel:
    def __init__(self, model_path: str = "best_pipeline.joblib", meta_path: Optional[str] = None):
        self.model_path = model_path
        self.meta_path = meta_path or (model_path + ".meta.json")
        self.bundle = None
        self.pre = None
        self.clf = None
        self.final_model = None
        self.meta = {}
        self.threshold = DEFAULT_THRESHOLD
        self.numeric_features = []
        self.categorical_features = []
        self.feature_cols = None
        self._load()

    def _load_meta(self):
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, "r") as f:
                    self.meta = json.load(f)
                # try to extract threshold
                if "cv_threshold" in self.meta:
                    try:
                        self.threshold = float(self.meta["cv_threshold"])
                    except Exception:
                        pass
                # extract lists
                self.numeric_features = self.meta.get("numeric_features", [])
                self.categorical_features = self.meta.get("categorical_features", [])
                self.feature_cols = self.numeric_features + self.categorical_features
            except Exception:
                self.meta = {}

    def _load(self):
        # load meta first
        self._load_meta()
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(self.model_path)
        self.bundle = joblib.load(self.model_path)
        # If bundle has direct predict_proba, treat as final model
        if hasattr(self.bundle, "predict_proba"):
            self.final_model = self.bundle
            return
        # If the bundle is a dict with components
        if isinstance(self.bundle, dict):
            # typical keys: 'pre', 'clf', 'meta', 'final_model' etc.
            self.pre = self.bundle.get("pre", None)
            self.clf = self.bundle.get("clf", None)
            self.final_model = self.bundle.get("final_model", None)
            # if 'meta' inside bundle, merge/override
            if "meta" in self.bundle and isinstance(self.bundle["meta"], dict):
                self.meta.update(self.bundle["meta"])
                self.threshold = float(self.meta.get("cv_threshold", self.threshold))
                self.numeric_features = self.meta.get("numeric_features", self.numeric_features)
                self.categorical_features = self.meta.get("categorical_features", self.categorical_features)
                self.feature_cols = self.numeric_features + self.categorical_features
            # If still no final_model, but have pre + clf -> we'll use them
            if self.final_model is None and (self.pre is not None and self.clf is not None):
                return
            if self.final_model is not None:
                return
        # fallback: if bundle is an sklearn Pipeline
        if hasattr(self.bundle, "named_steps") or hasattr(self.bundle, "steps"):
            # some pipelines provide predict_proba
            if hasattr(self.bundle, "predict_proba"):
                self.final_model = self.bundle
                return
            # else try to extract pre & clf
            try:
                self.pre = self.bundle.named_steps.get("pre", None)
                self.clf = self.bundle.named_steps.get("clf", None)
            except Exception:
                pass

    def _ensure_feature_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # If we have a feature_cols list from meta, select/order them and add missing columns with NaN
        if self.feature_cols:
            for c in self.feature_cols:
                if c not in df.columns:
                    df[c] = np.nan
            return df[self.feature_cols]
        # fallback: return df as-is
        return df

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        # For numeric_features: coerce to numeric (errors -> NaN)
        for c in self.numeric_features:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # For categorical: make sure dtype is string/object
        for c in self.categorical_features:
            if c in df.columns:
                df[c] = df[c].astype(str).fillna("missing")
        return df

    def _prep_dataframe(self, records: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[Any]]:
        if not isinstance(records, list):
            raise ValueError("records must be a list of dicts")
        df = pd.DataFrame(records)
        # retain patient ids for output (if available)
        patient_ids = df.get("patient_id", [None] * len(df)).tolist()
        # if meta lists available coerce types
        # If no meta, attempt to infer numeric columns by dtype
        if self.numeric_features or self.categorical_features:
            df = self._coerce_types(df)
        else:
            # try best-effort: convert object columns that look numeric
            for col in df.columns:
                if df[col].dtype == "object":
                    # try to coerce
                    df[col] = pd.to_numeric(df[col], errors="ignore")
            # now set numeric_features automatically (any numeric dtype)
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = [c for c in df.columns if c not in num_cols]
            self.numeric_features = num_cols
            self.categorical_features = cat_cols
            self.feature_cols = num_cols + cat_cols
        # ensure columns exist in expected order if available
        df = self._ensure_feature_columns(df)
        return df, patient_ids

    def predict_proba_for_records(self, records: List[Dict[str, Any]]) -> List[float]:
        """
        Accepts list-of-dicts. Returns list of positive-class probabilities.
        """
        if records is None or len(records) == 0:
            return []
        df, patient_ids = self._prep_dataframe(records)
        # If we have final_model with predict_proba, try directly.
        try:
            if self.final_model is not None and hasattr(self.final_model, "predict_proba"):
                # Some final_model may expect raw X (preprocessing included)
                X = df if (self.pre is None and not hasattr(self.final_model, "pre")) else df
                probs = self.final_model.predict_proba(X)
                proba_pos = probs[:, 1].astype(float).tolist()
                return proba_pos
        except Exception as e:
            # Fall through to other calls
            pass

        # If we have pre + clf
        if self.pre is not None and self.clf is not None:
            try:
                Xt = self.pre.transform(df)
            except Exception:
                # If pre is scikit-learn ColumnTransformer that expects numpy -> it should work.
                # If transform fails, try to fit a very simple transform: impute numeric -> fillna(0)
                Xt = df.copy()
                for c in self.numeric_features:
                    if c in Xt.columns:
                        Xt[c] = pd.to_numeric(Xt[c], errors="coerce").fillna(0)
                Xt = Xt.values
            probs = self.clf.predict_proba(Xt)
            return probs[:, 1].astype(float).tolist()

        # If clf alone (without pre), try passing df->clf
        if self.clf is not None and hasattr(self.clf, "predict_proba"):
            try:
                probs = self.clf.predict_proba(df)
                return probs[:, 1].astype(float).tolist()
            except Exception as e:
                raise RuntimeError(f"Model predict_proba failed: {e}")

        raise RuntimeError("No usable model found in the bundle for predict_proba()")

    def get_threshold(self) -> float:
        return float(self.threshold or DEFAULT_THRESHOLD)

    def prepare_prediction_record(self, prob: float, name: str = "Digital Twin Health Risk", patient_id: Optional[str] = None) -> Dict[str, Any]:
        thresh = self.get_threshold()
        pred = int(prob >= thresh)
        if prob >= 0.75:
            level = "High"
            desc = "High risk of abnormality"
        elif prob >= 0.5:
            level = "Medium"
            desc = "Moderate risk - monitor closely"
        else:
            level = "Low"
            desc = "Low risk"
        return {
            "risk_score": round(float(prob), 4),
            "prediction": pred,
            "alert_level": level,
            "description": desc,
            "name": name,
            "patient_id": patient_id if patient_id is not None else None,
            "updated_at": datetime.utcnow().isoformat()
        }
