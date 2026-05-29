# Estructura del proyecto

```
sentiment_project/
│
├── data/
│   ├── dataset_original.csv
│   ├── dataset_etiquetado.csv
│
├── models/
│
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

## Este proyecto contiene un `Pipeline` mínimo de `NLP`:

Explorar → Etiquetar → Preprocesar → Entrenar → Evaluar → Inferir.

# 01_exploracion.py

- Importa pandas
- Exploración inicial sobre dataset_original

# 02_etiquetado.py

- Importa pandas
- Carga dataset_original
- Genera una muestra de 120 comentarios aleatoria
- Envia la muestra a dataset_etiquetado

# Etiquetado manual de muestra
### Clases:
- positivo
- negativo
- neutro
```
positivo: 105
neutro:     9
negativo:   6
```
# 03_preprocesamiento.py
- Carga dataset_etiquetado
- Normaliza texto a minúscula
- Tratamiento de nulos a NaN
- Normaliza etiquetas a minúscula

# 04_entrenamiento.py
rutas robustas con Path
carga de dataset_etiquetado.csv
limpieza básica de texto
normalización de sentimiento
train_test_split con stratify
guardado de dataset_test.csv
entrenamiento con LogisticRegression(class_weight="balanced")
guardado de modelo y vectorizador con joblib
- Carga el dataset preprocesado
- Separa X e y
- Realiza train_test_split con stratify --> intenta mantener la proporción de positivo, neutro, negativo en train y test
- Vectorizacion con TfidfVectorizer 
- Entrena LogisticRegression con balanceo de clases

Luego de la ejecucion 
lee dataset_etiquetado.csv
limpia texto
divide train/test
vectoriza X_train
entrena LogisticRegression
termina

# 05_evaluacion.py
cargar modelo
cargar vectorizer
cargar dataset_test
transformar texto
predecir
comparar predicción vs etiqueta real
mostrar métricas
### Mide:
- accuracy
- precision
- recall
- F1
Agregar matriz de confusion

# 06_inferencia.py
- Aplica al dataset_original

