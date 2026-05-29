import pandas as pd

df = pd.read_csv("../data/dataset_etiquetado.csv")

df["texto"] = df["texto"].str.lower()

df["texto"] = df["texto"].fillna("")