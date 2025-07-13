import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

from src.config import FEATURES, TARGET_COL, MODEL_PATH

def train_model(df: pd.DataFrame) -> RandomForestClassifier:
    """
    Entrena un modelo Random Forest usando las columnas definidas en config.py.

    Parámetros:
    - df: DataFrame con features y columna objetivo

    Retorna:
    - modelo entrenado (RandomForestClassifier)
    """
    X = df[FEATURES]
    y = df[TARGET_COL]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model

def save_model(model: RandomForestClassifier, path: str = MODEL_PATH):
    """
    Guarda el modelo entrenado en formato joblib.

    Parámetros:
    - model: modelo entrenado
    - path: ruta donde guardar el archivo
    """
    joblib.dump(model, path)
