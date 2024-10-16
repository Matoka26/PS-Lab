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


def sinusoidal(a: float, f: int, t: float, phi: float) -> float:
    return a * np.sin(2 * np.pi * f * t + phi)


def inverse_fourier_transform(X: np.array, n: int) -> np.array:
    x = np.zeros(n, dtype=complex)

    for m in range(n):
        for k in range(n):
            x[m] += X[k] * np.exp(2j * np.pi * k * m / n)

    x /= n
    return x


if __name__ == '__main__':
    if not os.path.isdir(figures_directory1):
        os.makedirs(figures_directory1, exist_ok=True)

    if not os.path.isdir(figures_directory2):
        os.makedirs(figures_directory2, exist_ok=True)

    nof_samples = 2_000
    nof_components = nof_samples
    nof_sum_signals = 5

    # generate random signal
    frequencies = np.random.randint(size=nof_sum_signals, low=1, high=100, dtype=int)
    amplitudes = np.random.randint(size=nof_sum_signals, low=1, high=100, dtype=int)
    phases = np.zeros(nof_sum_signals)

    samples = np.linspace(0, 1, nof_samples)

    # sum up signal components
    wave = sinusoidal(amplitudes[0], frequencies[0], samples, phases[0])
    for i in range(1, nof_sum_signals):
        wave += sinusoidal(amplitudes[i], frequencies[i], samples, phases[i])

    # plot signal
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    fig.suptitle('Inverse Transform')
    ax[0].plot([0, 0], [max(wave), -max(wave)])
    ax[0].plot(samples, [0] * nof_samples)
    ax[0].plot(samples, wave)
    ax[0].set_xlabel('Time(s)')
    ax[0].set_ylabel('x(t)')

    # get the inverse
    inv = inverse_fourier_transform(wave, nof_components)
    freq = np.arange(0, nof_samples, nof_samples / nof_components)

    # plot inverse
    markerline, stemlines, baseline = ax[1].stem(
        freq, np.abs(inv), linefmt="k-", markerfmt="ko"
    )
    markerline.set_markerfacecolor("none")
    stemlines.set_linewidth(0.5)
    baseline.set_color("k")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("|X(ω)|")

    # cut the axes after reaching the max frequency and a bit more to look nice
    ax[1].set_xlim([0, max(frequencies) * 1.1])

    fig.savefig(f"./{figures_directory1}/ex3.pdf")
    fig.savefig(f"./{figures_directory2}/ex3.png")