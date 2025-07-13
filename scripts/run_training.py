# scripts/run_training.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.data.dataset import load_dataset, clean_data
from src.features.build_features import build_features
from src.modeling.train import train_model, save_model
from src.config import DATA_URL_JAN


def main():
    print("Cargando datos...")
    df_raw = load_dataset(DATA_URL_JAN)

    print("Limpiando datos...")
    df_clean = clean_data(df_raw)

    print("Generando variables...")
    df_final = build_features(df_clean)

    print("Entrenando modelo...")
    model = train_model(df_final)

    print("Guardando modelo...")
    save_model(model)

    print("✅ Entrenamiento completado y modelo guardado.")

if __name__ == "__main__":
    main()
