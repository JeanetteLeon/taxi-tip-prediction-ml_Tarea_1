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
    También guarda muestras (head 10) de los datos sin procesar y procesados.
    """

    # ---------------------
    # 1. Descargar datos crudos
    # ---------------------
    nombre_archivo = f"yellow_tripdata_{mes}.parquet"
    url = f"{PARQUET_BASE_URL}/{nombre_archivo}"
    ruta_local = os.path.join(TEST_RAW_PATH, nombre_archivo)

    if not os.path.exists(ruta_local):
        print(f"Descargando {mes} desde {url}...")
        os.makedirs(TEST_RAW_PATH, exist_ok=True)
        urllib.request.urlretrieve(url, ruta_local)
        print(f"✓ Archivo guardado en: {ruta_local}")
    else:
        print(f"{mes} ya fue descargado previamente.")

    # ---------------------
    # 2. Cargar y guardar muestra cruda
    # ---------------------
    df_raw = load_dataset(ruta_local)
    df_raw_sample = df_raw.head(10)
    df_raw_sample.to_parquet(
        os.path.join(TEST_RAW_PATH, f"sample_raw_{mes}.parquet"),
        index=False
    )

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
    df_final.to_parquet(
        os.path.join(TEST_PROCESSED_PATH, f"test_processed_{mes}.parquet"),
        index=False
    )

    # ---------------------
    # 5. Guardar head(10) procesado también en la carpeta oficial
    # ---------------------
    df_final.head(10).to_parquet(
        os.path.join(TEST_PROCESSED_PATH, f"sample_test_{mes}.parquet"),
        index=False
    )

    print(f"✓ Procesamiento y guardado completado para {mes}.\n")


if __name__ == "__main__":
    for mes in MESES_EVALUACION:
        descargar_y_procesar_mes(mes)
