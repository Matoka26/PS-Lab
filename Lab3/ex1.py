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

figures_directory = './figures'


def sinusoidal(a: float, f: int, t: float, phi: float) -> float:
    return a * np.cos(2 * np.pi * f * t + phi)


def get_fourier_matrix(x: np.array, n: int) -> np.ndarray:
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
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory, exist_ok=True)

    nof_samples = 2000
    n = 8

    frequencies = np.random.randint(size=n, low=1, high=100, dtype=int)
    amplitudes = np.random.randint(size=n, low=1, high=100, dtype=int)
    phases = np.zeros(n)

    samples = np.linspace(0, 1, nof_samples)

    # Create the signal
    wave = sinusoidal(amplitudes[0], frequencies[0], samples, phases[0])
    for comp in range(1, n):
        wave += sinusoidal(amplitudes[comp], frequencies[comp], samples, phases[comp])

    # plt.plot([0, 0], [max(wave), -max(wave)])
    # plt.plot(samples, [0] * nof_samples)
    # plt.plot(samples, wave)
    # plt.show()

    components = get_fourier_matrix(wave, n)

    print(f'Is unitary? {np.allclose(np.transpose(components), components)}')

    fig, axs = plt.subplots(n, figsize=(12, 6), sharex=True, sharey=True)
    fig.suptitle("Fourier Matrix")

    for i in range(n):
        axs[i].plot([j + 1 for j in range(n)], components[i].real)
        axs[i].plot([j + 1 for j in range(n)], components[i].imag, ".--")
        axs[i].set_ylabel(f"componenta {i}")

    fig.savefig(f"./{figures_directory}/ex1.pdf")