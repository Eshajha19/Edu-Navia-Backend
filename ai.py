import os
import random
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# ------------------- Configuration -------------------
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

dataset_path = "Dataset.csv"           # input dataset
out_path = "universities.csv"         # updated dataset
model_path = "models.pkl"             # saved models
encoder_path = "encoders.pkl"         # saved label encoders

# List of universities
universities = [
    "Indian Institute of Technology Delhi",
    "Indian Institute of Technology Bombay",
    "Indian Institute of Technology Madras",
    "Indian Institute of Technology Kanpur",
    "Indian Institute of Technology Kharagpur",
    "Indian Institute of Technology Roorkee",
    "Indian Institute of Technology Guwahati",
    "Indian Institute of Science Bangalore",
    "Jawaharlal Nehru University (JNU)",
    "Delhi University (DU)",
    "Jamia Millia Islamia",
    "Banaras Hindu University (BHU)",
    "Aligarh Muslim University",
    "University of Hyderabad",
    "Osmania University",
    "Jadavpur University",
    "Anna University",
    "Savitribai Phule Pune University",
    "Mumbai University",
    "Gujarat University",
    "Vellore Institute of Technology (VIT)",
    "Birla Institute of Technology & Science (BITS Pilani)",
    "SRM Institute of Science & Technology",
    "Manipal Academy of Higher Education",
    "Amity University",
    "Lovely Professional University (LPU)",
    "Chandigarh University",
    "Jain University",
    "Christ University",
    "Symbiosis International University",
    "National Institute of Technology Trichy",
    "National Institute of Technology Surathkal",
    "National Institute of Technology Warangal",
    "Indian Institute of Information Technology Allahabad",
    "Indian Institute of Information Technology Gwalior",
    "Birla Institute of Technology Mesra",
    "Thapar Institute of Engineering & Technology",
    "PSG College of Technology",
    "KIIT University",
    "Kalinga University",
    "Sharda University",
    "Galgotias University",
    "Graphic Era University",
    "Amrita Vishwa Vidyapeetham",
    "Dayananda Sagar University",
    "REVA University",
    "SR University",
    "KL University",
    "MIT World Peace University",
    "UPES Dehradun"
]

# ------------------- Helper -------------------
def create_demo_df(n=500):
    return pd.DataFrame({
        "score_10th": np.random.randint(60, 100, size=n),
        "score_12th": np.random.randint(55, 95, size=n),
        "entrance_score": np.random.randint(200, 800, size=n),
        "preferred_stream": np.random.choice(["Engineering","Science","Commerce","Arts","Law","Management"], size=n),
        "preferred_location": np.random.choice(["Delhi","Mumbai","Bangalore","Pune","Chennai","Hyderabad"], size=n),
        "budget_k_per_year": np.random.choice([50,100,150,200,300], size=n),
        "course": np.random.choice(["B.Tech","B.Sc","BBA","BA","B.Com","B.Arch","M.Tech","MBA"], size=n)
    })

# ------------------- Load Dataset -------------------
if os.path.exists(dataset_path):
    try:
        df = pd.read_csv(dataset_path)
        print(f"Loaded dataset from {dataset_path} with shape {df.shape}")
    except:
        print("Failed to load dataset. Creating demo data.")
        df = create_demo_df(500)
else:
    print("Dataset not found. Creating demo data.")
    df = create_demo_df(500)

# Add university column if missing
if "university" not in df.columns:
    df["university"] = np.random.choice(universities, size=len(df))
    print(f"Added 'university' column. Dataset shape: {df.shape}")

# Save updated dataset
df.to_csv(out_path, index=False)
print(f"Updated dataset saved to {out_path}")

# ------------------- Prepare Features & Targets -------------------
target_columns = []
if "university" in df.columns: target_columns.append("university")
if "course" in df.columns: target_columns.append("course")

feature_cols = [c for c in df.columns if c not in target_columns]
numeric_feats = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
cat_feats = [c for c in feature_cols if c not in numeric_feats]

# ------------------- Preprocessing -------------------
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_feats),
    ("cat", categorical_transformer, cat_feats)
])

# ------------------- Train Models -------------------
models = {}
label_encoders = {}

for target in target_columns:
    print(f"\nTraining model for target: {target}")
    X = df[feature_cols]
    y = df[target].astype(str)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    label_encoders[target] = le

    # Train on full dataset to avoid missing class issues
    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("xgb", XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=random_seed,
            num_class=len(le.classes_)
        ))
    ])
    clf.fit(X, y_enc)
    models[target] = clf

    # Print demo accuracy on same data
    y_pred = clf.predict(X)
    print(f"Accuracy for {target}: {accuracy_score(y_enc, y_pred):.4f}")

# ------------------- Save Models -------------------
with open(model_path, "wb") as f: pickle.dump(models, f)
with open(encoder_path, "wb") as f: pickle.dump(label_encoders, f)
print(f"Models saved to {model_path}")
print(f"Encoders saved to {encoder_path}")

# ------------------- Recommendation Function -------------------
def recommend(user_input: dict, top_k=5):
    row = {c: user_input.get(c, np.nan) for c in feature_cols}
    X_row = pd.DataFrame([row])
    results = {}
    for target in target_columns:
        clf = models[target]
        le = label_encoders[target]
        proba = clf.predict_proba(X_row)[0]
        top_idx = np.argsort(proba)[::-1][:top_k]
        results[target] = [(le.inverse_transform([i])[0], float(proba[i])) for i in top_idx]
    return results

# ------------------- Demo -------------------
example_input = {c: df.iloc[0][c] for c in feature_cols}
print("\nDemo input (first row):", example_input)
rec = recommend(example_input, top_k=5)
print("\nDemo recommendation:")
for target, vals in rec.items():
    print(target, "->")
    for name, prob in vals:
        print(f"  {name} ({prob:.3f})")
