# scripts/evaluate_months.py

import os
import pandas as pd
from src.config import MESES_EVALUACION, PARQUET_BASE_PATH
from src.data.dataset import load_dataset, clean_data
from src.features.build_features import build_features
from src.modeling.evaluation import load_model, evaluate_model

# Cargar modelo una vez
model = load_model()

# Lista para guardar resultados
resultados = []

# Evaluar por cada mes
for mes in MESES_EVALUACION:
    print(f"Evaluando mes: {mes}")

    # Construir ruta al archivo parquet local
    parquet_path = os.path.join(PARQUET_BASE_PATH, f"yellow_tripdata_{mes}.parquet")

    # Cargar y preparar los datos
    df = load_dataset(parquet_path)
    df = clean_data(df)
    df = build_features(df)

    # Evaluar modelo y guardar curva ROC
    plot_path = f"visualization/roc_curve_{mes}.png"
    metrics = evaluate_model(df, model, save_plot=True, plot_path=plot_path)

    # Guardar resultados
    resultados.append({
        "mes": mes,
        "cantidad_ejemplos": len(df),
        "f1_score": metrics["f1_score"],
        "accuracy": metrics["accuracy"],
        "roc_auc": metrics["roc_auc"]
    })

# Crear y guardar tabla de resultados
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("visualization/metrics_by_month.csv", index=False)
print("\nResultados guardados en visualization/metrics_by_month.csv")
