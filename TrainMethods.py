import numpy as np


def MSE(RealValue, PredictValue):
    return (RealValue - PredictValue) ** 2


def MAE(RealValue, PredictValue):
    return abs(RealValue - PredictValue)


# Method:
# MSE or MAE.
def LinearRegression(x, y, w, b, Method="MSE"):

    y_pred = ForwardPass(x, w, b)
    TotalData = len(y)

    if Method == "MSE":
        loss = np.sum(MSE(y, y_pred)) / TotalData
    else:
        loss = np.sum(MAE(y, y_pred)) / TotalData

    return loss / 2


def ForwardPass(x, w, b):
    return  x * w + b


def DerivateParams(x, y, w, b, Method="MSE"):

    y_pred = ForwardPass(x, w, b)
    TotalData = len(y)
    calcule = y - y_pred

    if Method == "MSE":
        db = -np.sum(calcule) / TotalData
        dw = -np.sum(calcule * x) / TotalData
    else:
        db = -np.sum(np.sign(calcule)) / TotalData
        dw = -np.sum(np.sign(calcule) * x) / TotalData


    return db, dw


def UpdateParams(w, dw, b, db, lr):
    updateW = w - lr * dw
    updateB = b - lr * db

    return updateW, updateB


def TrainModel(x, y, umbral, lr, max_iters = 1000, Method="MSE"):

    np.random.seed(42)

    # si no lo definimos en el () el tamaño, se vuelve escalar.
    w = np.random.randn()
    b = 0.0

    L = LinearRegression(x, y, w, b, Method)
    AvgLoss = []

    i = 0


    while L > umbral and i < max_iters:

        db, dw = DerivateParams(x, y, w, b, Method)
        w, b = UpdateParams(w, dw, b, db, lr)

        AvgLoss.append(L)
        L = LinearRegression(x, y, w, b, Method)

        i += 1
        if i % 20 == 0:
            print(f"iteracion: {i} --> loss:{L}")

    return b, w, np.mean(AvgLoss), AvgLoss, i