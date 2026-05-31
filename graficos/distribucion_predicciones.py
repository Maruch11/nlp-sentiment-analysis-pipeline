from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

GRAFICOS_DIR = BASE_DIR / "graficos"
GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

# Carga
modelo = joblib.load(MODELS_DIR / "modelo_sentimiento.pkl")
vectorizer = joblib.load(MODELS_DIR / "vectorizer.pkl")

df_test = pd.read_csv(DATA_DIR / "dataset_test.csv")

# Predicción
X_test = vectorizer.transform(df_test["texto"])

y_real = df_test["sentimiento"]
y_pred = modelo.predict(X_test)

# Conteos
etiquetas = ["negativo", "neutro", "positivo"]

real = y_real.value_counts().reindex(etiquetas, fill_value=0)
pred = pd.Series(y_pred).value_counts().reindex(etiquetas, fill_value=0)

# Gráfico
df_plot = pd.DataFrame({
    "real": real,
    "predicho": pred
})

ax = df_plot.plot(kind="bar")

plt.title("Distribución real vs predicha")
plt.xlabel("Clase")
plt.ylabel("Cantidad")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    GRAFICOS_DIR / "distribucion_predicciones.png",
    dpi=300
)

plt.close()

print("Gráfico exportado")