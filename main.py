import numpy as np


# Propagacion para adelante.
def ForwardPass(x, w, b):
    return x * w.T + b


def MSE(RealValue, PredictValue):
    return (RealValue - PredictValue) ** 2


def MAE(RealValue, PredictValue):
    return abs(RealValue - PredictValue)


# Method:
# MSE or MAE.
def LinearRegression(x, y, w, b, Method="MSE"):
    y_pred = ForwardPass(x, w, b)
    TotalData = len(x)
    s = 0
    Formula = MSE if Method == "MSE" else MAE

    for i in range(TotalData):
        s = s + Formula(y[i], y_pred[i])

    return s / (2 * TotalData)


def normalization(data):
    std_data = np.std(data)
    mean_data = np.mean(data)

    return (data - mean_data) / std_data


if __name__ == "__main__":
    DataSize = 1000
    np.random.seed(42)

    # Las horas estudiadas por estudiante...
    x = np.round(np.random.uniform(0, 10, DataSize), 2)

    # normalizar ese conjunto de datos
    x = normalization(x)

    # La nota que saco...
    y = np.random.randint(0, 20, DataSize)

    # inicializamos los pesos
    w = np.random.randn(DataSize)

    # inicializamos los sesgos
    b = np.zeros(DataSize)

    print(LinearRegression(x, y, w, b, "MSE"))
