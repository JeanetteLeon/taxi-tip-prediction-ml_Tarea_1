import pandas as pd

def load_dataset(path: str) -> pd.DataFrame:
    """
    Carga un archivo Parquet desde una URL o ruta local.

    Parámetros:
    - path: str -> URL o ruta local del archivo Parquet

    Retorna:
    - df: DataFrame con los datos cargados
    """
    df = pd.read_parquet(path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset y crea la variable 'tip', que vale 1 si se dejó propina (> 0), 0 en caso contrario.
    También elimina registros con tarifa o distancia no válida.

    Parámetros:
    - df: DataFrame original

    Retorna:
    - df: DataFrame limpio con la variable 'tip'
    """
    df = df.copy()
    df = df[(df["fare_amount"] > 0) & (df["trip_distance"] > 0)]
    df["tip"] = (df["tip_amount"] > 0).astype(int)
    return df
