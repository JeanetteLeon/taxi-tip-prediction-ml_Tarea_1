# src/modeling/evaluation.py

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix,
    roc_auc_score, roc_curve
)

# Importar constantes definidas en config.py (features, variable objetivo y ruta del modelo)
from src.config import FEATURES, TARGET_COL, MODEL_PATH


def load_model(path: str = MODEL_PATH):
    """
    Carga el modelo entrenado desde un archivo .joblib.
    """
    return joblib.load(path)


def evaluate_model(df: pd.DataFrame, model, save_plot=False, plot_path="visualization/roc_curve.png"):
    """
    Evalúa un modelo entrenado usando un conjunto de datos dado.
    
    Parámetros:
        df: DataFrame con los datos a evaluar.
        model: modelo RandomForest previamente entrenado.
        save_plot: si True, guarda la curva ROC como imagen.
        plot_path: ruta donde se guardará la curva ROC.

    Retorna:
        Un diccionario con las métricas de evaluación y la matriz de confusión.
    """

    # Selección de variables predictoras (X) y variable objetivo (y)
    X = df[FEATURES]
    y_true = df[TARGET_COL]

    # Predicción de clases y probabilidades
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]  # Probabilidades para la clase positiva

    # Cálculo de métricas de desempeño
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    # Creación del gráfico ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Línea de referencia
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')

    # Guardar gráfico si se indica
    if save_plot:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)

    plt.close()  # Cierra la figura para liberar memoria

    # Retorna métricas como diccionario
    return {
        "f1_score": f1,
        "accuracy": acc,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist()  # Convertida a lista para facilitar guardado o visualización
    }
