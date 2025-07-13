# scripts/download_months.py

import os
import urllib.request
import pandas as pd

from src.config import (
    MESES_EVALUACION,
    PARQUET_BASE_URL,
    TEST_RAW_PATH,
    TEST_PROCESSED_PATH
)
from src.data.dataset import load_dataset, clean_data
from src.features.build_features import build_features


def descargar_y_procesar_mes(mes: str):
    """
    Descarga, guarda crudo y procesa el Parquet mensual si no existe localmente.
    Guarda:
    - Parquet crudo completo en TEST_RAW_PATH
    - Muestra cruda (head 10) en data/raw/
    - Parquet procesado completo en TEST_PROCESSED_PATH
    - Muestra procesada (100k) en TEST_PROCESSED_PATH
    - Muestra procesada (head 10) en data/processed/test/
    """

    # Nombre del archivo
    nombre_archivo = f"yellow_tripdata_{mes}.parquet"
    url = f"{PARQUET_BASE_URL}/{nombre_archivo}"
    ruta_raw_completo = os.path.join(TEST_RAW_PATH, nombre_archivo)

    # ---------------------
    # 1. Descargar crudo si no existe
    # ---------------------
    if not os.path.exists(ruta_raw_completo):
        print(f"Descargando {mes} desde {url}...")
        os.makedirs(TEST_RAW_PATH, exist_ok=True)
        urllib.request.urlretrieve(url, ruta_raw_completo)
        print(f"✓ Archivo guardado en: {ruta_raw_completo}")
    else:
        print(f"{mes} ya fue descargado previamente.")

    # ---------------------
    # 2. Cargar y guardar muestra cruda (10 registros)
    # ---------------------
    df_raw = load_dataset(ruta_raw_completo)

    sample_raw_10_path = os.path.join("data", "raw", "test",  f"sample_raw_{mes}.csv")
    os.makedirs(os.path.dirname(sample_raw_10_path), exist_ok=True)
    df_raw.head(10).to_csv(sample_raw_10_path, index=False)

    # ---------------------
    # 3. Procesar datos
    # ---------------------
    print(f"Procesando datos del mes {mes}...")
    df_clean = clean_data(df_raw)
    df_final = build_features(df_clean)

    # ---------------------
    # 4. Guardar procesado completo
    # ---------------------
    os.makedirs(TEST_PROCESSED_PATH, exist_ok=True)
    processed_path = os.path.join(TEST_PROCESSED_PATH, f"test_processed_{mes}.parquet")
    df_final.to_parquet(processed_path, index=False)

    # ---------------------
    # 5. Guardar muestra procesada de 100k
    # ---------------------
    processed_100k_path = os.path.join(TEST_PROCESSED_PATH, f"sample_processed_100k_{mes}.parquet")
    df_final.head(100_000).to_parquet(processed_100k_path, index=False)

    # ---------------------
    # 6. Guardar muestra procesada de 10 registros
    # ---------------------
    sample_processed_10_path = os.path.join("data", "processed", "test", f"sample_test_{mes}.csv")
    os.makedirs(os.path.dirname(sample_processed_10_path), exist_ok=True)
    df_final.head(10).to_csv(sample_processed_10_path, index=False)

    print(f"✓ Procesamiento y guardado completado para {mes}.\n")


if __name__ == "__main__":
    for mes in MESES_EVALUACION:
        descargar_y_procesar_mes(mes)
