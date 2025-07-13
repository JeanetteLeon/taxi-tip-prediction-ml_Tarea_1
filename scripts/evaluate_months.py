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

# Crear carpetas necesarias
os.makedirs("visualization/roc_curve", exist_ok=True)
os.makedirs("visualization/conf_matrix", exist_ok=True)
os.makedirs("data/evaluation/predictions", exist_ok=True)

# Evaluar cada mes
for mes in MESES_EVALUACION:
    print(f"Evaluando mes: {mes}")

    # Ruta del archivo procesado
    parquet_path = os.path.join(TEST_PROCESSED_PATH, f"sample_processed_100k_{mes}.parquet")

    # Cargar datos procesados
    df = load_dataset(parquet_path)

    # Rutas para guardar las imágenes
    roc_path = f"visualization/roc_curve/roc_curve_{mes}.png"
    cm_path = f"visualization/conf_matrix/conf_matrix_{mes}.png"

    # Evaluar modelo y guardar imágenes
    metrics, _, df_preds = evaluate_model(
        df,
        model,
        save_plot=True,
        plot_path=roc_path,
        cm_path=cm_path
    )

    # Guardar predicciones por mes
    preds_path = f"data/evaluation/predictions/predictions_{mes}.csv"
    df_preds.to_csv(preds_path, index=False)

    # Guardar métricas
    resultados.append({
        "mes": mes,
        "cantidad_ejemplos": len(df),
        **metrics  # descompone el diccionario de métricas
    })

# Crear carpeta de salida para métricas si no existe
os.makedirs("data/evaluation", exist_ok=True)

# Guardar tabla de métricas
df_resultados = pd.DataFrame(resultados)
csv_path = "data/evaluation/metrics_by_month.csv"
df_resultados.to_csv(csv_path, index=False)

print(f"\n✓ Resultados guardados en {csv_path}")
