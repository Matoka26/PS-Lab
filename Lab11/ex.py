import os
import numpy as np
from utils import generate_time_serie, hankel_from_serie
import matplotlib.pyplot as plt

figures_dir = 'figures'


if __name__ == "__main__":
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir, exist_ok=True)

    N = 1000
    samples = np.arange(0, N)
    series = generate_time_serie(N)

    L = 5

    hankelized_series = hankel_from_serie(series, L)
    corelation_mat1 = np.matmul(hankelized_series, np.transpose(hankelized_series))
    corelation_mat2 = np.matmul(np.transpose(hankelized_series), hankelized_series)

    eig_values1 = np.linalg.eigvals(corelation_mat1)
    eig_values2 = np.linalg.eigvals(corelation_mat2)

    print(np.sqrt(eig_values2))

    U, S, V = np.linalg.svd(hankelized_series)

    print(S)
