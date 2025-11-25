import joblib
obj = joblib.load("best_pipeline.joblib")
print(type(obj))
print(obj)
