import numpy as np
from TrainMethods import TrainModel
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def normalization(data):
    std_data = np.std(data)
    mean_data = np.mean(data)
    return (data - mean_data) / (std_data + 1e-8)  # Evitar división por 0


def normalizationMinMax(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


if __name__ == "__main__":
    x_train = np.linspace(0, 50, 50)
    y_train = 12 * x_train + 5 + np.random.normal(0, 2, len(x_train))  # Datos con ruido

    # Normalización
    x_norm1 = (max(x_train) - x_train) /(max(x_train) - min(x_train))
    y_norm1 = (max(y_train) - y_train) /(max(y_train) - min(y_train))

    # Entrenamiento del modelo
    w, b, avgloss, AvgLossByIter, numIters = TrainModel(x_norm1, y_norm1, 0.001, 0.01,
                                                        max_iters=10 ** 3, Method="MAE")
    print(f"\nw: {w} \nb:{b}")

    # Predicción normalizada
    y_pred_norm = w * x_norm1 + b

    # Desnormalización correcta
    y_pred_original = np.max(y_train) - y_pred_norm * (np.max(y_train) - np.min(y_train))

    # Gráfico
    plt.scatter(x_train, y_train, label="Datos Reales", color="blue", alpha=0.6)
    plt.plot(x_train, y_pred_original, label=f"Modelo: {w:.2f}*x + {b:.2f}", color="red")

    plt.xlabel("Horas de estudio")
    plt.ylabel("Nota obtenida")
    plt.legend()
    plt.show()
