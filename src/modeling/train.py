import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

from src.config import FEATURES, TARGET_COL, MODEL_PATH

def train_model(df: pd.DataFrame) -> RandomForestClassifier:
    X = df[FEATURES]
    y = df[TARGET_COL]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model

def save_model(model: RandomForestClassifier, path: str = MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)  # 🔧 Asegura que el directorio exista
    joblib.dump(model, path)
