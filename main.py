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


if __name__ == "__main__":

    DataSize = 10 ** 5

    # Las horas estudiadas por estudiante...
    x = np.round(np.random.uniform(0, 10, DataSize), 2)

    # normalizar ese conjunto de datos
    x = normalization(x)



    # La nota que saco...
    y = np.random.randint(0, 20, DataSize)
    y = normalization(y)

    w, b, avgloss, AvgLossByIter, numIters = TrainModel(x, y, 0.045, 0.05,
                                                        max_iters= 10 ** 4, Method="MAE")

    print( avgloss )

    plt.plot(range(0, numIters), AvgLossByIter, marker='o', linestyle="-", color='blue', label='Loss')
    plt.ylim(0, 2 )
    plt.savefig('pe.png')
    plt.show()

