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
from utils import get_fourier_matrix, get_fourier_components

figures_directory1 = './figures_pdf'
figures_directory2 = './figures_png'


if __name__ == '__main__':
    if not os.path.isdir(figures_directory1):
        os.makedirs(figures_directory1, exist_ok=True)

    if not os.path.isdir(figures_directory2):
        os.makedirs(figures_directory2, exist_ok=True)

    n = 8

    F = get_fourier_matrix(n)
    Fh = np.transpose(np.conjugate(F))

    FhF = np.matmul(Fh, F)
    # Diag de 2 ori?????
    FhF = np.subtract(FhF, np.diag(np.diag(np.full((n, n), fill_value=FhF[0, 0]))))

    print(f'Is unitary? {np.allclose(np.linalg.norm(FhF, ord="fro"), 0)}')

    fig, axs = plt.subplots(n, figsize=(12, 6), sharex=True, sharey=True)
    fig.suptitle("Fourier Matrix")

    for i in range(n):
        axs[i].plot([j + 1 for j in range(n)], F[i].real)
        axs[i].plot([j + 1 for j in range(n)], F[i].imag, ".--")
        axs[i].set_ylabel(f"componenta {i}")

    fig.savefig(f"./{figures_directory1}/ex1.pdf")
    fig.savefig(f"./{figures_directory2}/ex1.png")