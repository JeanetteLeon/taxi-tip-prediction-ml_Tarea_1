# src/modeling/evaluation.py

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix,
    roc_auc_score, roc_curve
)

# Importar constantes definidas en config.py
from src.config import FEATURES, TARGET_COL, MODEL_PATH


def load_model(path: str = MODEL_PATH):
    """
    Carga el modelo entrenado desde un archivo .joblib.
    """
    return joblib.load(path)


def evaluate_model(
    df: pd.DataFrame,
    model,
    save_plot: bool = False,
    plot_path: str = "src/visualization/roc_curve.png",
    cm_path: str = None
):
    """
    Evalúa un modelo entrenado usando un conjunto de datos dado.

    Parámetros:
        df: DataFrame con los datos a evaluar.
        model: modelo RandomForest previamente entrenado.
        save_plot: si True, guarda la curva ROC como imagen.
        plot_path: ruta donde se guardará la curva ROC.
        cm_path: ruta donde se guardará la matriz de confusión (si se indica).

    Retorna:
        Un diccionario con las métricas de evaluación y la matriz de confusión.
    """
    # Variables predictoras y objetivo
    X = df[FEATURES]
    y_true = df[TARGET_COL]

    # Predicciones
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]  # Clase positiva

    # Métricas
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    # Gráfico ROC
    if save_plot:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Curva ROC')
        plt.legend(loc='lower right')
        plt.savefig(plot_path)
        plt.close()

    # Gráfico de matriz de confusión
    if cm_path:
        os.makedirs(os.path.dirname(cm_path), exist_ok=True)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Matriz de Confusión")
        plt.xlabel("Predicho")
        plt.ylabel("Real")
        plt.savefig(cm_path)
        plt.close()

    # Retorna métricas
    return {
        "f1_score": f1,
        "accuracy": acc,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist()
    }
