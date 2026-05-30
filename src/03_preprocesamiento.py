from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RUTA_PREPROCESADO = DATA_DIR / "dataset_preprocesado.csv"

df = pd.read_csv(DATA_DIR / "dataset_etiquetado.csv")

df["texto"] = df["texto"].fillna("")
df["texto"] = df["texto"].str.strip().str.lower()

df["sentimiento"] = df["sentimiento"].fillna("")
df["sentimiento"] = df["sentimiento"].str.strip().str.lower()

df = df[df["sentimiento"] != ""]

if RUTA_PREPROCESADO.exists():
    print(f"El archivo ya existe y no se va a sobrescribir: {RUTA_PREPROCESADO}")
else:
    df.to_csv(RUTA_PREPROCESADO, index=False)
    print(f"Archivo generado: {RUTA_PREPROCESADO}")
