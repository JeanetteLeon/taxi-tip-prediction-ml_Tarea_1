# scripts/download_months.py

import os
import urllib.request
import pandas as pd

from src.config import (
    MESES_EVALUACION,
    PARQUET_BASE_PATH,
    PARQUET_BASE_URL,
    TEST_RAW_PATH,
    TEST_PROCESSED_PATH
)
from src.data.dataset import load_dataset, clean_data
from src.features.build_features import build_features


def descargar_y_procesar_mes(mes: str):
    """
    Descarga, guarda crudo y procesa el Parquet mensual si no existe localmente.
    Guarda una muestra de 10 registros procesados como CSV.
    """

    # Ruta y URL para descarga completa
    url = f"{PARQUET_BASE_URL}/yellow_tripdata_{mes}.parquet"
    nombre_archivo = f"yellow_tripdata_{mes}.parquet"
    ruta_local = os.path.join(PARQUET_BASE_PATH, nombre_archivo)

    if not os.path.exists(ruta_local):
        print(f"Descargando {mes} desde {url}...")
        os.makedirs(PARQUET_BASE_PATH, exist_ok=True)
        urllib.request.urlretrieve(url, ruta_local)
        print(f"✓ Archivo guardado en: {ruta_local}")
    else:
        print(f"{mes} ya fue descargado previamente.")

    print(f"Procesando datos del mes {mes}...")

    # Cargar y procesar
    df_raw = load_dataset(ruta_local)
    df_clean = clean_data(df_raw)
    df_final = build_features(df_clean)

    # Guardar RAW reducido (HEAD 100.000)
    os.makedirs(TEST_RAW_PATH, exist_ok=True)
    df_raw.head(100000).to_parquet(
        os.path.join(TEST_RAW_PATH, f"test_raw_{mes}.parquet"),
        index=False
    )

    # Guardar procesado completo
    os.makedirs(TEST_PROCESSED_PATH, exist_ok=True)
    df_final.head(100000).to_parquet(
        os.path.join(TEST_PROCESSED_PATH, f"test_processed_{mes}.parquet"),
        index=False
    )

    # Guardar muestra de 10 registros procesados para revisión
    muestra_path = os.path.join("data", "processed", "test", f"sample_test_{mes}.csv")
    os.makedirs(os.path.dirname(muestra_path), exist_ok=True)
    df_final.head(10).to_csv(muestra_path, index=False)

    print(f"✓ Procesamiento y guardado completado para {mes}.\n")


if __name__ == "__main__":
    for mes in MESES_EVALUACION:
        descargar_y_procesar_mes(mes)
