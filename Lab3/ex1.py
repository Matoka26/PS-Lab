'''
Pentru N = 8 creat, i matricea Fourier. Desenat, i pe subplot-uri diferite
pentru fiecare linie partea real˘a s, i partea imaginar˘a. Verificat, i c˘a matricea Fourier F este unitar˘a (complex˘a s, i ortogonal˘a, adic˘a F
HF este
un multiplu al matricei identitate). Folosit, i funct, iile numpy.allclose sau
numpy.linalg.norm pentru a verifica unitaritatea.
'''

import matplotlib.pyplot as plt
import numpy as np
import os

figures_directory1 = './figures_pdf'
figures_directory2 = './figures_png'


def get_fourier_matrix(n: int) -> np.ndarray:
    F = np.zeros((n, n), dtype=complex)

    for k in range(n):
        for m in range(n):
            F[k, m] = np.exp(-2j * np.pi * k * m / n)
    return F


def get_fourier_components(x: np.array, nof_comp: int) -> np.array:
    N = len(x)
    X = np.zeros(nof_comp, dtype=complex)

    for m in range(nof_comp):
        sum = 0
        for k in range(nof_comp):
            sum += x[k] * np.exp(-2j * np.pi * k * m / N)
        X[m] = sum
    return X


if __name__ == '__main__':
    if not os.path.isdir(figures_directory1):
        os.makedirs(figures_directory1, exist_ok=True)

    if not os.path.isdir(figures_directory2):
        os.makedirs(figures_directory2, exist_ok=True)

    n = 8

    components = get_fourier_matrix(n)

    print(f'Is unitary? {np.allclose(np.transpose(components), components)}')

    fig, axs = plt.subplots(n, figsize=(12, 6), sharex=True, sharey=True)
    fig.suptitle("Fourier Matrix")

    for i in range(n):
        axs[i].plot([j + 1 for j in range(n)], components[i].real)
        axs[i].plot([j + 1 for j in range(n)], components[i].imag, ".--")
        axs[i].set_ylabel(f"componenta {i}")

    fig.savefig(f"./{figures_directory1}/ex1.pdf")
    fig.savefig(f"./{figures_directory2}/ex1.png")