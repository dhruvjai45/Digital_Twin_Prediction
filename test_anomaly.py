import joblib

m = joblib.load("anomaly_model.joblib")

print("Model type:", type(m))
print("Has decision_function:", hasattr(m, "decision_function"))
print("Has score_samples:", hasattr(m, "score_samples"))
print("Has predict:", hasattr(m, "predict"))
print("Has predict_proba:", hasattr(m, "predict_proba"))
