# scripts/train_model.py

import sys
import os
import pandas as pd

# Agrega el path del proyecto raíz para poder importar el módulo 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.dataset import load_dataset, clean_data
from src.features.build_features import build_features
from src.modeling.train import train_model, save_model
from src.config import DATA_URL_JAN, TRAIN_PROCESSED_PATH, TRAIN_RAW_PATH, MODEL_PATH


def main():
    print("Cargando datos...")

    # Cargar datos desde URL
    df_raw = load_dataset(DATA_URL_JAN)

    # --- Guardar dataset crudo completo (externo) ---
    os.makedirs(TRAIN_RAW_PATH, exist_ok=True)
    raw_filename = "yellow_tripdata_2020-01.parquet"
    raw_path = os.path.join(TRAIN_RAW_PATH, raw_filename)
    df_raw.to_parquet(raw_path, index=False)
    print(f"✓ Datos crudos guardados en: {raw_path}")

    # --- Guardar muestra cruda de 10 registros (proyecto) ---
    sample_raw_path = os.path.join("data", "raw", "train_sample_raw.csv")
    os.makedirs(os.path.dirname(sample_raw_path), exist_ok=True)
    df_raw.head(10).to_csv(sample_raw_path, index=False)

    print("Limpiando datos...")
    df_clean = clean_data(df_raw)

    print("Generando variables...")
    df_final = build_features(df_clean)

    # --- Guardar dataset procesado completo (externo) ---
    os.makedirs(TRAIN_PROCESSED_PATH, exist_ok=True)
    full_path = os.path.join(TRAIN_PROCESSED_PATH, "data_train.parquet")
    df_final.to_parquet(full_path, index=False)

    # --- Guardar muestra procesada de 100.000 registros (externo) ---
    sample_100k = df_final.head(100_000)
    sample_100k_path = os.path.join(TRAIN_PROCESSED_PATH, "train_sample_100k.parquet")
    sample_100k.to_parquet(sample_100k_path, index=False)

    # --- Guardar muestra procesada de 10 registros (proyecto) ---
    sample_10_path = os.path.join("data", "processed", "train", "train_sample.csv")
    os.makedirs(os.path.dirname(sample_10_path), exist_ok=True)
    df_final.head(10).to_csv(sample_10_path, index=False)

    print("Entrenando modelo...")
    model = train_model(sample_100k)

    print("Guardando modelo...")
    save_model(model, path=MODEL_PATH)

    print("✓ Proceso completado.")


if __name__ == "__main__":
    main()
