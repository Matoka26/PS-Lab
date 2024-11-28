import matplotlib.pyplot as plt
from utils import generate_time_serie
import os
import numpy as np

figures_dir = './figures'


def ma_model(serie: np.array, q: int) -> np.array:
    n = len(serie)
    train = serie[:n//0.8]
    test = serie[-n//0.2:]

    print(test)

    return np.array()


def get_estimation_error(serie: np.array, prediction: np.array) -> float:
    return np.sum([(serie[i-1]-prediction[i])**2 for i in range(1, len(serie))]) / len(serie)


if __name__ == "__main__":
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir, exist_ok=True)

    N = 1000
    samples = np.arange(0, N)
    serie = generate_time_serie(N)

    ma_model(serie)

    plt.savefig(f'{figures_dir}/ex3.pdf')

