import pandas as pd
import numpy as np


def ReadCSVData(CSVPath, Campos):
    # Forma más fácil para leer un CSV, lo almacena como un diccionario donde si ponemos
    # df[name_column], suelta todo el contenido que tiene esa columna, y adicionando df[name_column].tolist()
    # nos lo devuelve en formato lista.
    df = pd.read_csv(CSVPath)
    return np.array([df[CampoName].tolist() for CampoName in Campos])


def ObtainTrain(CSVPath, Campos):
    x = ReadCSVData(CSVPath, Campos)
    y = ReadCSVData(CSVPath, ["MTO_PIA"])

    return x, y


def ObtainTest(CSVPath, Campos):
    x = ReadCSVData(CSVPath, Campos)
    return x[:, 0:1464]
