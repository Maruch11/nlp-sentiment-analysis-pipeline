import pandas as pd
from /src/05_evaluacion.py import vectorizer, modelo
nuevos = pd.read_csv("../data/dataset_original.csv")

X = vectorizer.transform(
    nuevos["texto"]
)

pred = modelo.predict(X)

nuevos["prediccion"] = pred

print(
    nuevos[
        ["texto","prediccion"]
    ].head()
)
