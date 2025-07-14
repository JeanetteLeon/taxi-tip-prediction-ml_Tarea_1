
# taxi-tip-prediction-ml_Tarea_1  

**Reestructuración y Evaluación de Modelo de Machine Learning**

Este proyecto tiene como objetivo principal modularizar y evaluar un modelo de clasificación binaria que predice si una propina será alta o baja, utilizando datos de taxis en la ciudad de Nueva York. Se analizan diferentes meses del año 2020 para estudiar la evolución del rendimiento del modelo, especialmente durante el inicio de la pandemia por COVID-19.

---

## Estructura del Proyecto

Taxi-tip-prediction-ml_Tarea_1/
│
├── data/
│   ├── raw/          # Datos originales en muestras (datos completos están en local)
│   │   ├── train/
│   │   └── test/
│   └── processed/    # Datos preparados para modelamiento en muestras (datos completos están en local)
│       ├── train/
│       └── test/
│
├── notebooks/        # Notebook original y Notebook con análisis y evaluación del modelo
│
├── scripts/          # Scripts automatizados para ejecutar cada etapa del proceso del modelo
│
├── src/              # Código fuente del proyecto
│   ├── config/       # Rutas y constantes
│   ├── data/         # Funciones de carga y procesamiento
│   ├── modeling/     # Entrenamiento, predicción y evaluación
│   └── visualization/# Gráficos y análisis visual
│
├── requirements.txt  # Dependencias del proyecto
└── README.md         # Documentación general



---

## Objetivo

Evaluar la capacidad de generalización de un modelo de clasificación binaria de propinas entrenado en enero de 2020, contrastando su desempeño en los meses siguientes (febrero, marzo y abril), considerando los cambios provocados por la pandemia.

---

## Cómo ejecutar

1. **Instala las dependencias**:
   
   pip install -r requirements.txt


2. Configura las rutas:
Asegúrate de definir correctamente los paths en src/config/paths.py.


3. Ejecuta los scripts:
Puedes ejecutar los scripts de preprocesamiento, entrenamiento y evaluación desde la terminal (CMD) con los siguientes pasos:

        cd ruta/a/taxi-tip-prediction-ml_Tarea_1
        set PYTHONPATH=.
        python src/data/preprocess.py
        python src/modeling/train.py
        python src/modeling/evaluation.py

* Nota: Es importante establecer PYTHONPATH=. para que Python reconozca correctamente los módulos del proyecto al usar rutas relativas.

* Importante: Debido al volumen masivo de registros (millones de viajes), los datos de entrenamiento, testeo y el modelo entrenado fueron guardados localmente en el equipo y no se encuentran disponibles en este repositorio. Solo se trabaja con muestras representativas en la carpeta data/ y models/ para facilitar la reproducción del proyecto.


4. Visualización:
Los gráficos generados se guardan en src/visualization/.

## Dependencias principales
        pandas

        numpy

        scikit-learn

        matplotlib

        seaborn

        pyarrow

        (Para más detalles ver requirements.txt)



5. Análisis crítico y respuestas

Las respuestas y el análisis detallado del punto 4 de la tarea se encuentran en el siguiente notebook:

`notebooks/Analisis_critico_y_resultados.ipynb`

----
