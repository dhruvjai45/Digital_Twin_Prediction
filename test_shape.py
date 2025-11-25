import joblib

m = joblib.load("anomaly_model.joblib")

try:
    print("Anomaly model expects n_features_in_ =", m.n_features_in_)
except:
    print("Anomaly model has no n_features_in_ attribute")
