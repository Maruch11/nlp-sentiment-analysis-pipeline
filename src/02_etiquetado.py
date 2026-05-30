from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

ruta_original = BASE_DIR / "data" / "dataset_original.csv"
ruta_etiquetado = BASE_DIR / "data" / "dataset_etiquetado.csv"

# Cargar dataset
df = pd.read_csv(ruta_original)

# Generar una muestra reproducible de 120 comentarios aleatorios
df_muestra = df.sample(n=120, random_state=42)

# Enviar muestra a dataset_etiquetado solo si no existe
'''Se evita sobreescribir el csv en caso de reejecucion de este modulo, en atencion a que la etiquetacion es manual'''
if ruta_etiquetado.exists():
    print(f"El archivo ya existe y no se va a sobrescribir: {ruta_etiquetado}")
else:
    df_muestra.to_csv(ruta_etiquetado, index=False)
    print(f"Archivo generado: {ruta_etiquetado}")

# Resumen del dataset etiquetado
df_etiquetado = pd.read_csv(ruta_etiquetado)

print("\nResumen del dataset etiquetado:")
print(f"Filas: {len(df_etiquetado)}")

# Distribución de etiquetas
if "sentimiento" in df_etiquetado.columns:
    print("\nDistribución de etiquetas:")
    print(df_etiquetado["sentimiento"].value_counts(dropna=False))