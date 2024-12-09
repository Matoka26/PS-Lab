import matplotlib.pyplot as plt
from utils import generate_time_serie
import os
import numpy as np

figures_dir = './figures'


def ma_model(serie: np.array, q: int) -> np.array:
    n = len(serie)
    mean = np.mean(serie)
    serie = serie - mean
    b = serie[q:]
    e = np.random.normal(0, 1, n)
    e = np.concatenate(([np.mean(serie)], e))

    A = []
    for i in range(0, n-q):
        A.append(e[i: i+q])

    A = np.array(A)
    thetas, res, _, _ = np.linalg.lstsq(A, b)

    return np.dot(thetas, b[-len(thetas):]) + mean


def get_estimation_error(serie: np.array, prediction: np.array) -> float:
    return np.sum([(serie[i-1]-prediction[i])**2 for i in range(1, len(serie))]) / len(serie)


if __name__ == "__main__":
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir, exist_ok=True)

    N = 1000
    q = 5
    m = 20
    samples = np.arange(0, N)
    serie = generate_time_serie(N)

    predictions = []
    for i in range(N - m):
        predictions.append(ma_model(serie[i: i+m], q))

    plt.plot(samples, serie, label='Series')
    plt.plot(samples[m:], predictions, label='Prediction')
    plt.title('MA model prediction')
    plt.grid()
    plt.legend()
    plt.savefig(f'{figures_dir}/ex3.pdf')

