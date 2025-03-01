import numpy as np
from TrainMethods import TrainModel
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from ReadCSV import ObtainXandY


def normalization(data):
    std_data = np.std(data)
    mean_data = np.mean(data)
    return (data - mean_data) / (std_data + 1e-8)  # Evitar división por 0


def normalizationMinMax(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


if __name__ == "__main__":

    Campos = ["CANT_META_ANUAL", "CANT_META_SEM", "AVAN_FISICO_ANUAL", "AVAN_FISICO_SEM"]
    CsvPath = "./Data/train.csv"

    x_train, y_train = ObtainXandY(CsvPath, Campos)

    # Normalización
    x_norm1 = (np.max(x_train) - x_train) /(np.max(x_train) - np.min(x_train))
    y_norm1 = (np.max(y_train) - y_train) /(np.max(y_train) - np.min(y_train))

    # NORM PIOLA
    x_norm2 = normalization(x_train)
    y_norm2 = normalization(y_train)

    # Entrenamiento del modelo
    w, b, avgloss, AvgLossByIter, numIters = TrainModel(x_norm2, y_norm2, 0.001, 0.01,
                                                        max_iters=10 ** 4, Method="MAE")
    print(f"\nw: {w} \nb:{b}")