# scripts/download_months.py

import os
import urllib.request
from src.config import MESES_EVALUACION, PARQUET_BASE_PATH, PARQUET_BASE_URL

def descargar_parquet(mes: str):
    """
    Descarga el archivo Parquet para el mes indicado, si no existe localmente.
    """
    url = f"{PARQUET_BASE_URL}/yellow_tripdata_{mes}.parquet"
    ruta_local = os.path.join(PARQUET_BASE_PATH, f"yellow_tripdata_{mes}.parquet")

    if not os.path.exists(ruta_local):
        print(f"Descargando {mes} desde {url}...")
        os.makedirs(PARQUET_BASE_PATH, exist_ok=True)
        urllib.request.urlretrieve(url, ruta_local)
        print(f"Archivo guardado en: {ruta_local}")
    else:
        print(f"{mes} ya fue descargado previamente.")

if __name__ == "__main__":
    for mes in MESES_EVALUACION:
        descargar_parquet(mes)
