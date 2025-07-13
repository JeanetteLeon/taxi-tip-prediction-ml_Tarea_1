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

# --- Dataset fuente para entrenamiento (enero 2020) ---
DATA_URL_JAN = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-01.parquet"

# --- Datos procesados ---
TRAIN_PROCESSED_PATH = (
    "C:\\Users\\jeane_bkpplgv\\OneDrive\\Escritorio\\UDD\\5° Trimestre\\Desarrollo de Proyectos y Productos de Datos\\tarea_1\\data\\processed\\train\\train_data"
)

# --- Ruta donde se guardarán los datos crudos (sin procesar) de test ---
TEST_RAW_PATH = (
    "C:\\Users\\jeane_bkpplgv\\OneDrive\\Escritorio\\UDD\\5° Trimestre\\Desarrollo de Proyectos y Productos de Datos\\tarea_1\\data\\raw"
)

# --- Ruta donde se guardarán los datos procesados de test ---
TEST_PROCESSED_PATH = (
    "C:\\Users\\jeane_bkpplgv\\OneDrive\\Escritorio\\UDD\\5° Trimestre\\Desarrollo de Proyectos y Productos de Datos\\tarea_1\\data\\processed\\test"
)


# --- Modelo entrenado ---
MODEL_PATH = (
    "C:\\Users\\jeane_bkpplgv\\OneDrive\\Escritorio\\UDD\\5° Trimestre\\Desarrollo de Proyectos y Productos de Datos\\tarea_1\\models\\model.joblib"
)

# --- URL base de descarga para los meses de evaluación ---
PARQUET_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"

# --- Meses a evaluar ---
MESES_EVALUACION = ['2020-02', '2020-03', '2020-04']
