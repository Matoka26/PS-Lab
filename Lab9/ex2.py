import matplotlib.pyplot as plt
from utils import generate_time_serie
import os
import numpy as np

figures_dir = './figures'


def exponential_averaging(x: np.array, alpha: float) -> np.array:
    # if not 0 < alpha < 1:
    #     raise ValueError("Alpha must be between 0 and 1")

    n = len(x)
    new_serie = np.zeros(n)

    new_serie[0] = x[0]
    for t in range(1, n):
        new_serie[t] = alpha * x[t] + (1-alpha) * new_serie[t-1]

    return new_serie


def get_estimation_error(serie: np.array, prediction: np.array) -> float:
    return np.sum([(serie[i-1]-prediction[i])**2 for i in range(1, len(serie))]) / len(serie)


if __name__ == "__main__":
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir, exist_ok=True)

    N = 1000
    eps = 1e-6
    alphas = np.linspace(0, 1, 300)

    samples = np.arange(0, N)
    serie = generate_time_serie(N)
    errors = []

    best_err = np.infty
    best_alpha = -1
    for alpha in alphas:
        ea_serie = exponential_averaging(serie, alpha)

        err = get_estimation_error(serie, ea_serie)
        if best_err > err:
            best_err = err
            best_alpha = alpha
        errors.append(err)

    ea_serie = exponential_averaging(serie, best_alpha)

    plt.plot(samples, serie, label='Initial')
    plt.plot(samples + 1, ea_serie, label=f'Best averaging with alpha {"{:.3f}".format(best_alpha)}')

    plt.grid(True)
    plt.legend()
    plt.savefig(f'{figures_dir}/ex2.pdf')

    plt.clf()
    err_samples = np.linspace(0, 1, len(errors))
    plt.plot(err_samples, errors)
    plt.scatter(err_samples[np.argmin(errors)], np.min(errors), c='r')
    plt.yscale("log")
    plt.grid(True)
    plt.savefig(f'{figures_dir}/ex2_error.pdf')