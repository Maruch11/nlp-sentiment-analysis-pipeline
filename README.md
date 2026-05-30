# Sentiment Project

Pipeline de NLP para análisis de sentimiento.

## Requisitos

pip install -r requirements.txt

# Estructura del proyecto

```
nlp-sentiment-analysis-pipeline/
│
├── data/
│   ├── dataset_original.csv
│   ├── dataset_etiquetado.csv
|   ├── dataset_prerocesado.csv
│   ├── dataset_test.csv
|
├── models/
│   ├── modelo_sentimiento.pkl
│   ├── vectorizer.pkl
|
├── src/
│   ├── 01_exploracion.py
│   ├── 02_etiquetado.py
│   ├── 03_preprocesamiento.py
│   ├── 04_entrenamiento.py
│   ├── 05_evaluacion.py
│   └── 06_inferencia.py
│
├── requirements.txt
│
└── README.md
```

## Este proyecto contiene un `Pipeline` mínimo de `NLP`
```
Explorar → Etiquetar → Preprocesar → Entrenar → Evaluar → Inferir
```
### Flujo de datos
```
dataset_original.csv
        ↓
dataset_etiquetado.csv
        ↓
dataset_preprocesado.csv
(dataset limpio previo al entrenamiento)
        ↓
train_test_split
(train 96 / test 24)
        ↓
dataset_test.csv
(reserva del 20 % utilizada únicamente para evaluación)
        ↓
modelo_sentimiento.pkl + vectorizer.pkl
        ↓
inferencia sobre dataset_original.csv
```
## 01_exploracion.py

Responsable de realizar una exploración inicial del dataset original.

### Funcionalidad

- Carga `dataset_original.csv`
- Realiza exploración inicial de la estructura y contenido del dataset
- Inspecciona columnas, dimensiones, tipos de datos y valores faltantes

### Librerías principales

- pandas
- pathlib.Path

## 02_etiquetado.py

Responsable de generar una muestra para etiquetado manual.

### Funcionalidad

- Carga `dataset_original.csv`
- Genera una muestra aleatoria de 120 comentarios
- Exporta la muestra a `dataset_etiquetado.csv`

### Librerías principales

- pandas
- pathlib.Path

### Etiquetado manual

Clases utilizadas:

- positivo
- neutro
- negativo

Distribución del dataset etiquetado manualmente (120 registros):

| Clase | Cantidad |
|------|------:|
| positivo | 105 |
| neutro | 9 |
| negativo | 6 |

## 03_preprocesamiento.py

Responsable de normalizar y preparar los datos para entrenamiento.

### Funcionalidad

- Carga `dataset_etiquetado.csv`
- Normaliza texto y etiquetas (lowercase + eliminación de espacios)
- Reemplaza valores nulos en texto y sentimiento
- Elimina registros sin etiqueta de sentimiento
- Genera `dataset_preprocesado.csv`

### Librerías principales

- pandas
- pathlib.Path

## 04_entrenamiento.py

Responsable de entrenar el modelo de análisis de sentimiento a partir del dataset preprocesado.

### Funcionalidad

- Carga `dataset_preprocesado.csv`
- Separa variables predictoras (`X`) y etiquetas (`y`)
- Realiza división train/test usando split estratificado para mantener la proporción de clases
- Guarda `dataset_test.csv`
- Vectoriza texto utilizando `TfidfVectorizer`
- Entrena un modelo `LogisticRegression(class_weight="balanced")`
- Guarda modelo entrenado y vectorizador usando `joblib`

### Librerías principales

- pandas
- pathlib.Path
- sklearn.model_selection.train_test_split
- sklearn.feature_extraction.text.TfidfVectorizer
- sklearn.linear_model.LogisticRegression
- joblib

### Artefactos generados

- `dataset_test.csv`
- modelo serializado
- vectorizador serializado

## 05_evaluacion.py

Responsable de evaluar el desempeño del modelo entrenado.

### Funcionalidad

- Carga modelo entrenado y vectorizador
- Carga `dataset_test.csv`
- Transforma texto utilizando el vectorizador
- Realiza predicciones sobre el conjunto de prueba
- Compara predicciones contra etiquetas reales
- Calcula métricas de evaluación

### Métricas evaluadas

- Accuracy
- Precision
- Recall
- F1-score
- Matriz de confusión

### Librerías principales

- pandas
- pathlib.Path
- joblib
- sklearn.metrics

## 06_inferencia.py

Responsable de aplicar el modelo entrenado sobre datos no vistos.

### Funcionalidad

- Carga modelo entrenado y vectorizador
- Carga `dataset_original.csv`
- Transforma texto utilizando el vectorizador
- Genera predicciones de sentimiento sobre el dataset original
- Exporta resultados inferidos

### Librerías principales

- pandas
- pathlib.Path
- joblib

## Criterios prácticos de reejecución de módulos

| Artefacto | Origen | Acción |
|---|---|---|
| `dataset_original.csv` | Dataset fuente / generado manualmente | No sobrescribir |
| `dataset_etiquetado.csv` | Resultado del etiquetado manual | No sobrescribir |
| `dataset_preprocesado.csv` | Resultado del preprocesamiento | No sobrescribir salvo decisión explícita |
| `dataset_test.csv` | Resultado del entrenamiento | Sobrescribir en cada entrenamiento |
| `modelo_sentimiento.pkl` | Modelo entrenado | Sobrescribir en cada entrenamiento |
| `vectorizer.pkl` | Vectorizador entrenado | Sobrescribir en cada entrenamiento |

### Regla general

- Artefactos manuales → preservar
- Artefactos intermedios reproducibles → regenerar cuando corresponda
- Modelo y artefactos de entrenamiento → reemplazar en cada nuevo entrenamiento
