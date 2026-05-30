from pathlib import Path
import sys

import joblib
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

modelo = joblib.load(MODELS_DIR / "modelo_sentimiento.pkl")
vectorizer = joblib.load(MODELS_DIR / "vectorizer.pkl")

nuevos = pd.read_csv(DATA_DIR / "dataset_original.csv")
nuevos["texto"] = nuevos["texto"].fillna("").str.lower()

X = vectorizer.transform(nuevos["texto"])
pred = modelo.predict(X)

nuevos["prediccion"] = pred

print(nuevos[["texto", "prediccion"]].head())
