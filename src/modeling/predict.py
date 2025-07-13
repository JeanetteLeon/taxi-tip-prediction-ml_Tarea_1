import os
import pandas as pd
from src.config import MESES_EVALUACION, TEST_PROCESSED_PATH # scripts/evaluate_months.py

import os
import pandas as pd
from src.config import MESES_EVALUACION, TEST_PROCESSED_PATH
from src.data.dataset import load_dataset
from src.modeling.predict import load_model, evaluate_model
from src.data.dataset import load_dataset, clean_data
from src.features.build_features import build_features
from src.modeling.predict import load_model, evaluate_model

# Cargar modelo una vez
model = load_model()

# Lista para guardar resultados
resultados = []

# Evaluar por cada mes
for mes in MESES_EVALUACION:
    print(f"Evaluando mes: {mes}")

    # Construir ruta al archivo parquet local
    parquet_path = os.path.join(TEST_PROCESSED_PATH, f"yellow_tripdata_{mes}.parquet")

    # Cargar y preparar los datos
    df = load_dataset(parquet_path)
    df = clean_data(df)
    df = build_features(df)

    # Definir rutas para guardar las imágenes
    roc_path = f"src/visualization/roc_curve_{mes}.png"
    cm_path = f"src/visualization/conf_matrix_{mes}.png"

    # Evaluar modelo
    metrics, _ = evaluate_model(
        df,
        model,
        save_plot=True,
        plot_path=roc_path,
        cm_path=cm_path
    )

    # Guardar resultados en lista
    resultados.append({
        "mes": mes,
        "cantidad_ejemplos": len(df),
        "f1_score": metrics["f1_score"],
        "accuracy": metrics["accuracy"],
        "roc_auc": metrics["roc_auc"]
    })

# Crear y guardar tabla de resultados
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("src/visualization/metrics_by_month.csv", index=False)
print("\nResultados guardados en src/visualization/metrics_by_month.csv")
