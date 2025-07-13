# scripts/evaluate_months.py

import os
import pandas as pd
from src.config import MESES_EVALUACION, TEST_PROCESSED_PATH
from src.data.dataset import load_dataset
from src.modeling.predict import load_model, evaluate_model

# Cargar modelo una sola vez
model = load_model()

# Lista para guardar métricas por mes
resultados = []

# Evaluar cada mes
for mes in MESES_EVALUACION:
    print(f"Evaluando mes: {mes}")

    # Ruta del archivo procesado
    parquet_path = os.path.join(TEST_PROCESSED_PATH, f"test_processed_{mes}.parquet")

    # Cargar datos procesados (ya limpios y con features)
    df = load_dataset(parquet_path)

    # Evaluar modelo y guardar curva ROC
    os.makedirs("visualization", exist_ok=True)
    plot_path = f"visualization/roc_curve_{mes}.png"
    metrics = evaluate_model(df, model, save_plot=True, plot_path=plot_path)

    # Guardar métricas
    resultados.append({
        "mes": mes,
        "cantidad_ejemplos": len(df),
        "f1_score": metrics["f1_score"],
        "accuracy": metrics["accuracy"],
        "roc_auc": metrics["roc_auc"]
    })

# Crear carpeta de salida si no existe
os.makedirs("data/evaluation", exist_ok=True)

# Crear y guardar tabla de resultados
df_resultados = pd.DataFrame(resultados)
csv_path = "data/evaluation/metrics_by_month.csv"
df_resultados.to_csv(csv_path, index=False)
print(f"\nResultados guardados en {csv_path}")

