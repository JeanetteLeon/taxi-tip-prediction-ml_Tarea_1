# src/config.py

# --- Features ---
NUMERIC_FEATURES = [
    "pickup_weekday", "pickup_hour", "work_hours", "pickup_minute",
    "passenger_count", "trip_distance", "trip_time", "trip_speed"
]

CATEGORICAL_FEATURES = ["PULocationID", "DOLocationID", "RatecodeID"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --- Target ---
TARGET_COL = "high_tip"
TIP_THRESHOLD = 0.2  # > 20% del valor del viaje

# --- Constantes ---
EPS = 1e-7

# --- Dataset (ruta por defecto) ---
# Puedes reemplazar por otra URL si usas otro mes
DATA_URL_JAN = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-01.parquet"

# --- Modelo entrenado ---
MODEL_PATH = "C:\\Users\\jeane_bkpplgv\\OneDrive\\Escritorio\\UDD\\5° Trimestre\\Desarrollo de Proyectos y Productos de Datos\\tarea_1\\model_outputs\\model.joblib"


# --- Meses para evaluación del modelo ---
MESES_EVALUACION = ['2020-01', '2020-02', '2020-03']



