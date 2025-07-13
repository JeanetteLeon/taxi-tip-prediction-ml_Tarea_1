
from src.config import FEATURES, TARGET_COL, TIP_THRESHOLD, EPS

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica ingeniería de características y crea la variable objetivo `high_tip`.

    Parámetros:
    - df: DataFrame con los datos crudos (de enero, febrero, etc.)

    Retorna:
    - df: DataFrame con features finales + columna target (`high_tip`)
    """
    df = df.copy()

    # Crear columna de fracción de propina
    df["tip_fraction"] = df["tip_amount"] / df["fare_amount"]
    df[TARGET_COL] = (df["tip_fraction"] > TIP_THRESHOLD)

    # Crear variables de tiempo
    df["pickup_weekday"] = df["tpep_pickup_datetime"].dt.weekday
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_minute"] = df["tpep_pickup_datetime"].dt.minute

    # Variable booleana: si fue en horario laboral (lunes a viernes de 8 a 18)
    df["work_hours"] = (
        (df["pickup_weekday"] >= 0) & (df["pickup_weekday"] <= 4) &
        (df["pickup_hour"] >= 8) & (df["pickup_hour"] <= 18)
    )

    # Calcular duración y velocidad del viaje
    df["trip_time"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds()
    df["trip_speed"] = df["trip_distance"] / (df["trip_time"] + EPS)

    # Filtrar columnas de interés y convertir a float32
    columnas_finales = FEATURES + [TARGET_COL]
    df = df[columnas_finales].astype("float32").fillna(-1.0)

    # Convertir la columna objetivo a int32 (0 o 1)
    df[TARGET_COL] = df[TARGET_COL].astype("int32")

    return df.reset_index(drop=True)