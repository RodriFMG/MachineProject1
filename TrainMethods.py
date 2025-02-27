import numpy as np


def MSE(RealValue, PredictValue):
    return (RealValue - PredictValue) ** 2


def MAE(RealValue, PredictValue):
    return abs(RealValue - PredictValue)


def ForwardPass(x, w, b):
    return x * w + b


def LinearRegression(x, y, w, b, Method="MSE", lambda_L1=0, lambda_L2=0):
    y_pred = ForwardPass(x, w, b)
    TotalData = len(y)

    if Method == "MSE":
        loss = np.sum(MSE(y, y_pred)) / TotalData
    else:
        loss = np.sum(MAE(y, y_pred)) / TotalData

    # Aplicar regularización L1 y L2
    loss += lambda_L1 * np.sum(np.abs(w)) + (lambda_L2 / 2) * np.sum(w ** 2)

    return loss / 2


def DerivateParams(x, y, w, b, Method="MSE", lambda_L1=0, lambda_L2=0):
    y_pred = ForwardPass(x, w, b)
    TotalData = len(y)
    error = y - y_pred

    if Method == "MSE":
        db = -np.sum(error) / TotalData
        dw = -np.sum(error * x) / TotalData
    else:
        db = -np.sum(np.sign(error)) / TotalData
        dw = -np.sum(np.sign(error) * x) / TotalData

    # Aplicar derivadas de regularización
    dw += lambda_L2 * w + lambda_L1 * np.sign(w)  # L2 usa w, L1 usa signo(w)

    return db, dw


def UpdateParams(w, dw, b, db, lr):
    w -= lr * dw
    b -= lr * db
    return w, b


def TrainModel(x, y, umbral, lr, max_iters=1000, Method="MSE", lambda_L1=0, lambda_L2=0):
    np.random.seed(42)
    w = np.random.randn()
    b = 0.0

    L = LinearRegression(x, y, w, b, Method, lambda_L1, lambda_L2)
    AvgLoss = []
    i = 0

    while L > umbral and i < max_iters:
        db, dw = DerivateParams(x, y, w, b, Method, lambda_L1, lambda_L2)
        w, b = UpdateParams(w, dw, b, db, lr)

        AvgLoss.append(L)
        L = LinearRegression(x, y, w, b, Method, lambda_L1, lambda_L2)

        i += 1
        if i % 20 == 0:
            print(f"Iteración: {i} --> Pérdida: {L:.5f}")

    return b, w, np.mean(AvgLoss), AvgLoss, i
