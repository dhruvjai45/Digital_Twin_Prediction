import joblib

bundle = joblib.load("best_pipeline.joblib")
pre = bundle["pre"]

print("\nNUMERIC COLUMNS USED DURING TRAINING:\n", pre.transformers_[0][2])
print("\nCATEGORICAL COLUMNS USED DURING TRAINING:\n", pre.transformers_[1][2])
