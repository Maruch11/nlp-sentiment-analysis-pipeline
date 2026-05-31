from pathlib import Path
from transformers import pipeline

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_PATH = BASE_DIR / "models" / "huggingface"

# Crear directorio si no existe
MODELS_PATH.mkdir(parents=True, exist_ok=True)

clf = pipeline(
    task="text-classification",
    model="pysentimiento/robertuito-sentiment-analysis",
    cache_dir=str(MODELS_PATH)
)

texto = "Me encantó esta publicación"

resultado = clf(texto)

print(resultado)