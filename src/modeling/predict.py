# src/modeling/predict.py

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, log_loss, balanced_accuracy_score,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

from src.config import FEATURES, TARGET_COL


def load_model(path: str = "models/model.joblib"):
    """
    Carga el modelo previamente entrenado desde disco.
    """
    return joblib.load(path)


def evaluate_model(df, model, save_plot=False, plot_path=None, cm_path=None):
    """
    Evalúa el modelo y opcionalmente guarda la curva ROC y la matriz de confusión.

    Retorna:
    - dict con métricas
    - y_pred (predicciones binarias)
    - df_preds: DataFrame con columnas: true_label, predicted_label, probability
    """
    X = df[FEATURES]
    y_true = df[TARGET_COL]
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    # Calcular métricas
    metrics = {
        "f1_score": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_proba),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }

    # Guardar curva ROC
    if save_plot and plot_path:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {metrics['roc_auc']:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    # Guardar matriz de confusión
    if save_plot and cm_path:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()

    # DataFrame con valores reales y predichos
    df_preds = pd.DataFrame({
        "true_label": y_true.values,
        "predicted_label": y_pred,
        "probability": y_proba
    })

    return metrics, y_pred, df_preds
