from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

modelo = joblib.load(MODELS_DIR / "modelo_sentimiento.pkl")
vectorizer = joblib.load(MODELS_DIR / "vectorizer.pkl")

df_test = pd.read_csv(DATA_DIR / "dataset_test.csv")

X_test = vectorizer.transform(df_test["texto"])
y_test = df_test["sentimiento"]

pred = modelo.predict(X_test)

print(classification_report(y_test, pred))