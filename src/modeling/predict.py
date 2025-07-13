import pandas as pd
import joblib
from sklearn.metrics import f1_score

from src.config import FEATURES, TARGET_COL, MODEL_PATH

def load_model(path: str = MODEL_PATH):
    """
    Carga un modelo desde archivo.

    Parámetros:
    - path: ruta al archivo .joblib del modelo

    Retorna:
    - modelo cargado
    """
    return joblib.load(path)

def evaluate_model(model, df: pd.DataFrame) -> float:
    """
    Evalúa el modelo usando F1-score.

    Parámetros:
    - model: modelo entrenado
    - df: DataFrame con features + columna objetivo

    Retorna:
    - F1-score (float)
    """
    X = df[FEATURES]
    y_true = df[TARGET_COL]

    y_pred = model.predict(X)
    score = f1_score(y_true, y_pred)

    return score
