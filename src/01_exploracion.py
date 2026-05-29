from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

ruta_original = BASE_DIR / "data" / "dataset_original.csv"
ruta_etiquetado = BASE_DIR / "data" / "dataset_etiquetado.csv"

# Cargar dataset
df = pd.read_csv(ruta_original)

# Verificar estructura
print(df.head())
print(df.info())
print(df.shape)
print(df.columns)
print(df.isnull().sum())

