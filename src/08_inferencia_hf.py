from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models" / "huggingface"

# usar cache local del proyecto
os.environ["HF_HOME"] = str(MODELS_DIR)

import pandas as pd
from transformers import pipeline

DATA_DIR = BASE_DIR / "data"

clf = pipeline(
    task="text-classification",
    model="pysentimiento/robertuito-sentiment-analysis"
)

# cargar dataset original
df = pd.read_csv(
    DATA_DIR / "dataset_original.csv"
)

# inferencia
df["pred_hf"] = df["texto"].apply(
    lambda x: clf(str(x))[0]["label"]
)

# guardar resultados
df.to_csv(
    DATA_DIR / "dataset_inferido_hf.csv",
    index=False
)

print(f"Predicciones realizadas: {len(df)}")
print("\nCantidad por etiqueta:")
print(df["pred_hf"].value_counts())

print(df[df["pred_hf"]=="POS"][["texto", "pred_hf"]].head())

print(df[df["pred_hf"]=="NEU"][["texto", "pred_hf"]].head())

print(df[df["pred_hf"]=="NEG"][["texto", "pred_hf"]].head())
