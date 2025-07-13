import sys
import os
import pandas as pd

# Agrega el path del proyecto raíz para poder importar el módulo 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.dataset import load_dataset, clean_data
from src.features.build_features import build_features
from src.modeling.train import train_model, save_model
from src.config import DATA_URL_JAN, TRAIN_PROCESSED_PATH, MODEL_PATH


def main():
    print("Cargando datos...")
    df_raw = load_dataset(DATA_URL_JAN)

    print("Limpiando datos...")
    df_clean = clean_data(df_raw)

    print("Generando variables...")
    df_final = build_features(df_clean)

    # --- Guardar versión externa (100000 registros) ---
    df_to_save = df_final.head(100000)
    os.makedirs(os.path.dirname(TRAIN_PROCESSED_PATH), exist_ok=True)
    df_final.to_parquet(f"{TRAIN_PROCESSED_PATH}data_train.parquet", index=False)

    # --- Guardar muestra interna para revisión rápida (10 registros) ---
    sample_path = "data/processed/train/train_sample.csv"
    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
    df_final.head(10).to_csv(sample_path, index=False)

    print("Entrenando modelo...")
    model = train_model(df_to_save)

    print("Guardando modelo...")
    save_model(model, path=MODEL_PATH)

    print("Proceso completado.")


if __name__ == "__main__":
    main()
