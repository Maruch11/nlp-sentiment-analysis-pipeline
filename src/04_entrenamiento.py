from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Cargar df 
df = pd.read_csv(DATA_DIR / "dataset_etiquetado.csv")

# Normalizacion y tratamiento de espacios y textos vacios 
df["texto"] = df["texto"].fillna("")
df["texto"] = df["texto"].str.lower()

df["sentimiento"] = df["sentimiento"].str.strip().str.lower()

# Eliminacion de filas sin etiqueta
df = df.dropna(subset=["sentimiento"])

# train_test_split con stratify
X_train,X_test,y_train,y_test = train_test_split(
    df["texto"],
    df["sentimiento"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentimiento"]
)

# Guardar datos del test
df_test = pd.DataFrame({
    "texto": X_test.values,
    "sentimiento": y_test.values
})

df_test.to_csv(DATA_DIR / "dataset_test.csv", index=False)

# Vectorizacion
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)

# Modelo con balanceo de clases
modelo = LogisticRegression(class_weight="balanced", max_iter=1000)

modelo.fit(X_train,y_train)

# Guardar modelo y vectorizador
joblib.dump(modelo, MODELS_DIR / "modelo_sentimiento.pkl")
joblib.dump(vectorizer, MODELS_DIR / "vectorizer.pkl")

