# scripts/predict.py

# src/modeling/predict.py

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
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
    - predicciones del modelo
    """
    X = df[FEATURES]
    y_true = df[TARGET_COL]
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    # Calcular métricas
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)

    # Guardar curva ROC
    if save_plot and plot_path:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
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

    return {
        "f1_score": f1,
        "accuracy": acc,
        "roc_auc": roc_auc
    }, y_pred
