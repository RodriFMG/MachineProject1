import numpy as np
from TrainMethods import TrainModel

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def normalization(data):
    std_data = np.std(data)
    mean_data = np.mean(data)

    # Para evitar divison entre 0, pues se le suma un numero extremadamente pequeño.
    return (data - mean_data) / (std_data + 1e-8)

def normalizationMinMax(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

if __name__ == "__main__":


    # Las horas estudiadas por estudiante...
    x = np.linspace(0, 10, 50)

    # normalizar ese conjunto de datos
    x_norm1 = x / np.max(x)
    x_norm2 = normalization(x)
    x_norm3 = normalizationMinMax(x)


    # La nota que saco...
    y = 3 * x + 5 + np.random.normal(0, 2, len(x))
    y_norm1 = y / np.max(y)
    y_norm2 = normalization(y)
    y_norm3 = normalizationMinMax(y)

    w, b, avgloss, AvgLossByIter, numIters = TrainModel(x_norm3, y_norm3, 0.001, 0.025,
                                                        max_iters= 10 ** 4, Method="MSE")

    # predicción lineal del conjunto de datos

    print(f"w: {w} \n b:{b}")
    y_pred_norm = w * x_norm1 + b
    y_pred_original = y_pred_norm * (np.max(y) - np.min(y)) + np.min(y)
    #y_pred_original = y_pred_norm * np.max(y)

    print( avgloss )

    # scatter srive para poner una serie de puntos en la gráfica, pero estos no serán partes del plot
    # como tal
    plt.scatter(x, y, label = "Datos Reales",color = "blue", alpha=0.6)
    plt.plot(x, y_pred_original, label = f"Modelo: {w}*x + {b}")

    plt.xlabel("Variable independiente")
    plt.ylabel("Variable dependiente")

    plt.show()


