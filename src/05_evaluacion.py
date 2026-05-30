from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

modelo = joblib.load(MODELS_DIR / "modelo_sentimiento.pkl")
vectorizer = joblib.load(MODELS_DIR / "vectorizer.pkl")

df_test = pd.read_csv(DATA_DIR / "dataset_test.csv")

if df_test.empty:
    raise ValueError("Dataset test vacío")

columnas = ["texto", "sentimiento"]

if not all(c in df_test.columns for c in columnas):
    raise ValueError("Faltan columnas requeridas")

X_test = vectorizer.transform(df_test["texto"])
y_test = df_test["sentimiento"]

y_pred = modelo.predict(X_test)

# Mostrar primero la distribución casos reales hay por clase y cuántos predijo el modelo
print("="*20)
print("INFORMES")
print("="*20)

print("\n" + "-"*20)
print("Distribución real:")
print("-"*20)
print(y_test.value_counts())

print("\n" + "-"*20)
print("Distribución predicha:")
print("-"*20)
print(pd.Series(y_pred).value_counts())

# Matriz de confusion
etiquetas = ["negativo", "neutro", "positivo"]

matriz = confusion_matrix(y_test, y_pred, labels=etiquetas)

df_matriz = pd.DataFrame(
    matriz,
    index=[f"real_{e}" for e in etiquetas],
    columns=[f"pred_{e}" for e in etiquetas]
)

print("\n" + "-"*20)
print("Matriz de confusión:")
print("-"*20)
print(df_matriz)

# Reporte
print("\n" + "-"*20)
print("Reporte:")
print("-"*20)
# print(classification_report(y_test, y_pred)) # con warnings
print(classification_report(y_test, y_pred, zero_division=0)) # evita warnings cuando una clase no fue predicha