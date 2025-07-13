# scripts/evaluate_months.py

import pandas as pd
from src.config import MODEL_PATH, MESES_EVALUACION
from src.data.dataset import load_dataset, clean_data
from src.features.build_features import build_features
from src.modeling.evaluation import load_model, evaluate_model

# Cargar modelo entrenado
model = load_model(MODEL_PATH)

# Lista para guardar resultados
resultados = []

# Evaluación para cada mes
for mes in MESES_EVALUACION:
    print(f"Evaluando mes: {mes}")

    # Cargar y preparar datos del mes
    df = load_dataset(mes)
    df = clean_data(df)
    df = build_features(df)

    # Definir rutas para guardar visualizaciones
    roc_path = f"src/visualization/roc_curve_{mes}.png"
    cm_path = f"src/visualization/conf_matrix_{mes}.png"

    # Evaluar modelo y guardar visualizaciones
    metrics = evaluate_model(df, model, save_plot=True, plot_path=roc_path, cm_path=cm_path)

    # Agregar resultados a la lista
    resultados.append({
        "mes": mes,
        "cantidad_ejemplos": len(df),
        "f1_score": metrics["f1_score"],
        "accuracy": metrics["accuracy"],
        "roc_auc": metrics["roc_auc"]
    })

# Convertir resultados a DataFrame y mostrar
df_resultados = pd.DataFrame(resultados)
print("\nResultados de evaluación por mes:")
print(df_resultados)

# Guardar tabla en CSV
df_resultados.to_csv("src/visualization/resultados_evaluacion_mensual.csv", index=False)
