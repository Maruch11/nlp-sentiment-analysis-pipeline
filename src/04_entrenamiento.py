from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Cargar df 
df = pd.read_csv(DATA_DIR / "dataset_preprocesado.csv")

if df.empty:
    raise ValueError("Dataset vacío")

# Validar columnas requeridas
columnas = ["texto", "sentimiento"]

if not all(c in df.columns for c in columnas):
    raise ValueError("Faltan columnas requeridas")

# train_test_split con stratify
X_train_texto, X_test_texto, y_train, y_test = train_test_split(
    df["texto"],
    df["sentimiento"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentimiento"]
)

# Guardar datos del test
df_test = pd.DataFrame({
    "texto": X_test_texto,
    "sentimiento": y_test
})

df_test.to_csv(DATA_DIR / "dataset_test.csv", index=False)

# Vectorizacion
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train_texto)

# Modelo con balanceo de clases
modelo = LogisticRegression(class_weight="balanced", max_iter=1000)

modelo.fit(X_train, y_train)

# Guardar modelo y vectorizador
joblib.dump(modelo, MODELS_DIR / "modelo_sentimiento.pkl")
joblib.dump(vectorizer, MODELS_DIR / "vectorizer.pkl")

# Confirmación de carga del dataset
print("Dataset cargado:")
print(df.shape)

# Distribución de clases antes del split
print("\nDistribución de clases:")
print(df["sentimiento"].value_counts())

# Tamaño de train y test
print("\nTamaño de entrenamiento:")
print(len(X_train_texto))

print("\nTamaño de test:")
print(len(X_test_texto))

# Distribución de clases en train y test
print("\nDistribución en entrenamiento:")
print(y_train.value_counts())

print("\nDistribución en test:")
print(y_test.value_counts())

# Confirmación de guardado
print("\nModelo guardado en:")
print(MODELS_DIR / "modelo_sentimiento.pkl")

print("\nVectorizador guardado en:")
print(MODELS_DIR / "vectorizer.pkl")

print("\nDataset de test guardado en:")
print(DATA_DIR / "dataset_test.csv")