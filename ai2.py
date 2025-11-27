from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import pickle
import pandas as pd
import os

app = FastAPI()

# Allow frontend JS requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained models and encoders
with open("models.pkl", "rb") as f:
    models = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)  # only for target labels

# Extract feature columns from trained model pipeline
first_model = list(models.values())[0]
feature_cols = first_model.named_steps["preprocessor"].transformers_[0][2] + \
               first_model.named_steps["preprocessor"].transformers_[1][2]

# Serve static frontend files
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static\ai.html")

@app.post("/recommend")
async def recommend(request: Request):
    try:
        data = await request.json()

        # Build input row with defaults for missing fields
        row = {col: data.get(col, None) for col in feature_cols}
        X_row = pd.DataFrame([row])

        results = {}

        for target, model in models.items():
            le = label_encoders[target]

            # Predict probabilities
            proba = model.predict_proba(X_row)[0]

            # Top 5 recommendations
            top = proba.argsort()[::-1][:5]
            results[target] = [
                {"label": le.inverse_transform([i])[0], "probability": float(proba[i])}
                for i in top
            ]

        return JSONResponse(content={"status": "success", "recommendations": results})

    except Exception as e:
        print("Error:", e)
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
