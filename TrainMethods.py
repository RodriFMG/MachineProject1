import numpy as np


def MSE(RealValue, PredictValue):
    return (RealValue - PredictValue) ** 2


def MAE(RealValue, PredictValue):
    return abs(RealValue - PredictValue)


def ForwardPass(x, w, b):
    # @ igual que np.dot()
    return x @ np.transpose(w) + b


def LinearRegression(x, y, w, b, Method="MSE", lambda_L1=0, lambda_L2=0):
    y_pred = ForwardPass(x, w, b)
    TotalData = len(y)

    if Method == "MSE":
        loss = np.mean(MSE(y, y_pred))
    else:
        loss = np.mean(MAE(y, y_pred))

    # Aplicar regularización L1 y L2
    loss += lambda_L1 * np.sum(np.abs(w)) + (lambda_L2 / 2) * np.sum(w ** 2)

    return loss / 2


def DerivateParams(x, y, w, b, Method="MSE", lambda_L1=0, lambda_L2=0):

    y_pred = ForwardPass(x, w, b)

    TotalData = len(y)
    error = y - y_pred

    if Method == "MSE":
        db = -np.mean(error)
        dw = -np.mean(error * x)
    else:
        db = -np.mean(np.sign(error))
        dw = -np.mean(np.sign(error) * x)

    # Aplicar derivadas de regularización
    dw += (lambda_L2 / len(y)) * w
    dw += lambda_L1 * np.sign(w)

    return db, dw


def UpdateParams(w, dw, b, db, lr):

    w -= lr * dw
    b -= lr * db

    return w, b


def TrainModel(x, y, umbral, lr, max_iters=1000, Method="MSE", lambda_L1=0, lambda_L2=0):

    np.random.seed(42)

    numColumns = x.shape[1]
    numFilas = x.shape[0]

    w = np.random.rand(1, numColumns)
    b = np.random.rand(numFilas, 1)

    L = LinearRegression(x, y, w, b, Method, lambda_L1, lambda_L2)
    AvgLoss = []
    i = 0

    while L > umbral and i < max_iters:
        db, dw = DerivateParams(x, y, w, b, Method, lambda_L1, lambda_L2)
        w, b = UpdateParams(w, dw, b, db, lr)

        AvgLoss.append(L)
        L = LinearRegression(x, y, w, b, Method, lambda_L1, lambda_L2)

        i += 1
        if i % 10 == 0:
            print(f"Iteración: {i} --> Pérdida: {L:.5f}")

    return w, b, np.mean(AvgLoss), AvgLoss, i
