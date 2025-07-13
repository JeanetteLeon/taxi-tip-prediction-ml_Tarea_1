# scripts/download_months.py

import os
import urllib.request
import pandas as pd

from src.config import (
    MESES_EVALUACION,
    PARQUET_BASE_PATH,
    PARQUET_BASE_URL,
    FEATURES,
    TARGET_COL
)
from src.data.dataset import load_dataset, clean_data
from src.features.build_features import build_features


def descargar_y_procesar_mes(mes: str):
    """
    Descarga y procesa el Parquet mensual si no existe localmente.
    Guarda una muestra de 10 registros para revisión.
    """
    # Ruta y URL
    url = f"{PARQUET_BASE_URL}/yellow_tripdata_{mes}.parquet"
    ruta_local = os.path.join(PARQUET_BASE_PATH, f"yellow_tripdata_{mes}.parquet")

    if not os.path.exists(ruta_local):
        print(f"Descargando {mes} desde {url}...")
        os.makedirs(PARQUET_BASE_PATH, exist_ok=True)
        urllib.request.urlretrieve(url, ruta_local)
        print(f"Archivo guardado en: {ruta_local}")
    else:
        print(f"{mes} ya fue descargado previamente.")

    # Procesar
    print(f"Procesando datos del mes {mes}...")
    df_raw = load_dataset(ruta_local)
    df_clean = clean_data(df_raw)
    df_final = build_features(df_clean)

    # Guardar .parquet completo
    parquet_proc_path = os.path.join(PARQUET_BASE_PATH, f"processed_{mes}.parquet")
    df_final.to_parquet(parquet_proc_path, index=False)

    # Guardar muestra de 10 registros en CSV para inspección
    muestra_path = os.path.join("data", "processed", f"sample_{mes}.csv")
    os.makedirs(os.path.dirname(muestra_path), exist_ok=True)
    df_final.head(10).to_csv(muestra_path, index=False)

    print(f"✓ Procesamiento y guardado completado para {mes}.\n")


if __name__ == "__main__":
    for mes in MESES_EVALUACION:
        descargar_y_procesar_mes(mes)
