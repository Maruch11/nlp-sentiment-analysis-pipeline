from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

ruta_original = BASE_DIR / "data" / "dataset_original.csv"
ruta_etiquetado = BASE_DIR / "data" / "dataset_etiquetado.csv"

# Cargar dataset
df = pd.read_csv(ruta_original)

# Generar una muestra de 120 comentarios aleatorios
df_muestra = df.sample(n=120, random_state=42)

# Enviar muestra a dataset_etiquetado
df_muestra.to_csv(ruta_etiquetado, index=False)
