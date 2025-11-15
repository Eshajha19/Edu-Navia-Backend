from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pickle
import pandas as pd

app = FastAPI()

# CORS setup
origins = [
    "http://127.0.0.1:5500",   # local frontend
    "http://localhost:5500",    # local alternative
    "https://edu-navia.netlify.app",  # deployed frontend
    "*"  # allow all origins
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained models and encoders
with open("models.pkl", "rb") as f:
    models = pickle.load(f)
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# POST /recommend endpoint
@app.post("/recommend")
async def recommend(request: Request):
    try:
        data = await request.json()
        # Fill missing columns with defaults
        row = {
            "score_10th": data.get("score_10th", 0),
            "score_12th": data.get("score_12th", 0),
            "entrance_score": data.get("entrance_score", 0),
            "preferred_stream": data.get("preferred_stream", "Unknown"),
            "preferred_location": data.get("preferred_location", "Unknown"),
            "budget_k_per_year": data.get("budget_k_per_year", 0),
            "course": data.get("course", "Unknown"),
            "gender": "Other",
            "marks": 0,
            "stream": "Unknown",
            "subject": "Unknown"
        }

        X_row = pd.DataFrame([row])

        # Encode categorical features safely
        for col in ["preferred_stream", "preferred_location", "course", "gender", "stream", "subject"]:
            if col in encoders and col in X_row.columns:
                le = encoders[col]
                if X_row[col][0] in le.classes_:
                    X_row[col] = le.transform(X_row[col])
                else:
                    X_row[col] = 0  # fallback for unseen values

        results = {}
        for target in models:
            clf = models[target]
            le = encoders[target]
            proba = clf.predict_proba(X_row)[0]
            top_idx = proba.argsort()[::-1][:5]
            results[target] = [(le.inverse_transform([i])[0], float(proba[i])) for i in top_idx]

        return JSONResponse(content=results)

    except Exception as e:
        print("Error:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

