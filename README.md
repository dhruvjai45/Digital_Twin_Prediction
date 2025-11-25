# Digital Twin Prediction API

FastAPI service that:
- loads an XGBoost readmission-risk model and a preprocessor,
- receives patient vitals and returns readmission prob + AI-generated clinical descriptions via Groq.

## To run locally

1. Copy `.env.example` to `.env` and fill values
2. (Optional) upload .joblib models and set MODEL_URL and PREPROCESSOR_URL
3. Install:
   pip install -r requirements.txt
4. Run:
   uvicorn app:app --reload

## Deploy
Recommended: Deploy on Render as a Web Service. See deployment instructions in repo.