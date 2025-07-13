# scripts/run_training.py

import sys
import os

# Agrega el path del proyecto raíz para poder importar el módulo 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


from scr.data.dataset import load_dataset, clean_data
from scr.features.build_features import build_features
from scr.modeling.train import train_model, save_model
from scr.config import DATA_URL_JAN


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

    print("Proceso completado.")

if __name__ == "__main__":
    main()
