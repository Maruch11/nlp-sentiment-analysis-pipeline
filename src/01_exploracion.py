from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

ruta_original = BASE_DIR / "data" / "dataset_original.csv"
ruta_etiquetado = BASE_DIR / "data" / "dataset_etiquetado.csv"

# Cargar dataset
df = pd.read_csv(ruta_original)

# Verificar estructura
print("head:")
print(df.head())
print("\ninfo:")
print(df.info())
print("\nshape:")
print(df.shape)
print("\ncolumns:")
print(df.columns)
print("\nnulos:")
print(df.isnull().sum())

# Comentarios duplicados
print("\nComentarios duplicados:")
print(df["texto"].duplicated().sum())

textos_unicos = df["texto"].nunique(dropna=False)
duplicados_sin_original = df["texto"].duplicated().sum()
filas_con_texto_repetido = df["texto"].duplicated(keep=False).sum()
textos_repetidos_distintos = (df["texto"].value_counts(dropna=False) > 1).sum()

print("\nResumen de duplicados:")
print(f"{textos_repetidos_distintos} = textos distintos que aparecen mas de una vez")
print(f"{duplicados_sin_original} = duplicados extra, sin contar la primera aparicion")
print(f"{filas_con_texto_repetido} = filas involucradas en duplicados, incluyendo originales")
print(f"{textos_unicos} = textos unicos")

# Estadistica de comentarios
df["longitud_texto"] = df["texto"].fillna("").str.len()

print("\nEstadisticas de comentarios:")
print(df["longitud_texto"].describe())
