import os
import pandas as pd
import numpy as np
import pickle
from compress_pickle import load  # For models.lz4
from fastapi import FastAPI
from pydantic import BaseModel

# ------------------- Configuration -------------------
dataset_path = "admission_prediction_full_dataset.csv"
model_path = "models.lz4"      # compressed model
encoder_path = "encoders.pkl"  # uncompressed encoder

# ------------------- Load Dataset -------------------
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ Dataset not found: {dataset_path}")

df = pd.read_csv(dataset_path)
print(f"Loaded dataset with shape: {df.shape}")

# ------------------- Prepare Features -------------------
target_columns = ["university", "course"]
feature_cols = [c for c in df.columns if c not in target_columns]

# ------------------- Load Models and Encoders -------------------
if not os.path.exists(model_path) or not os.path.exists(encoder_path):
    raise FileNotFoundError("❌ models.lz4 or encoders.pkl not found.")

print("🔄 Loading models and encoders ...")
models = load(model_path, compression="lz4")  # compressed
with open(encoder_path, "rb") as f:
    label_encoders = pickle.load(f)
print("✔ Models and encoders loaded successfully!")

# ------------------- FastAPI Setup -------------------
app = FastAPI(title="EduNavia Recommendation API")

class UserInput(BaseModel):
    __root__: dict  # Accept arbitrary feature dict

@app.post("/recommend")
def recommend_endpoint(user_input: UserInput, top_k: int = 5):
    user_data = user_input.__root__
    row = {c: user_data.get(c, np.nan) for c in feature_cols}
    X_row = pd.DataFrame([row])
    results = {}

    for target in target_columns:
        clf = models[target]
        le = label_encoders[target]
        proba = clf.predict_proba(X_row)[0]
        top_idx = np.argsort(proba)[::-1][:top_k]
        results[target] = [
            {"name": le.inverse_transform([i])[0], "probability": float(proba[i])}
            for i in top_idx
        ]

    return results

# ------------------- Optional Test Example -------------------
if __name__ == "__main__":
    example_input = {c: df.iloc[0][c] for c in feature_cols}
    print("\n📌 Example prediction:")
    preds = {}
    row = pd.DataFrame([example_input])
    for target in target_columns:
        clf = models[target]
        le = label_encoders[target]
        proba = clf.predict_proba(row)[0]
        top_idx = np.argsort(proba)[::-1][:5]
        preds[target] = [(le.inverse_transform([i])[0], float(proba[i])) for i in top_idx]

    for target, values in preds.items():
        print(f"\n🎯 Top predicted {target}:")
        for name, prob in values:
            print(f"  → {name} ({prob:.3f})")

