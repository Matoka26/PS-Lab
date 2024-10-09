'''
1. Generat, i un semnal sinusoidal de tip sinus, de amplitudine, frecvent,˘a s, i
faz˘a aleas˘a de voi. Generat, i apoi un semnal de tip cosinus astfel ˆıncˆat pe
orizontul de timp ales, acesta s˘a fie identic cu semnalul sinus. Verificat, i
afis,ˆandu-le grafic ˆın dou˘a subplot-uri diferite.
'''

import os
import numpy as np
import matplotlib.pyplot as plt

figures_directory = './figures'


def sinusoidal(a: float, f: int, t: float, phi: float) -> float:
    return a * np.sin(2 * np.pi * f * t + phi)


def cosinusoidal(a: float, f: int, t: float, phi: float) -> float:
    return a * np.cos(2 * np.pi * f * t + phi - np.pi/2)


if __name__ == "__main__":
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory, exist_ok=True)

    fig, ax = plt.subplots(nrows=2, ncols=1)

    nof_samples = 1000
    f = 10
    a = 2
    phi = np.pi

    # plot sin
    samples = np.linspace(0, 1, nof_samples)
    ax[0].plot(samples, sinusoidal(a, f, samples, phi), c='b')
    ax[0].plot([0, 0], [a, -a])
    ax[0].plot(samples, [0] * nof_samples)
    ax[0].set_xlabel("Figure 1: Sin signal wave")

    # plot cos
    ax[1].plot(samples, cosinusoidal(a, f, samples, phi), c='r')
    ax[1].plot([0, 0], [a, -a])
    ax[1].plot(samples, [0] * nof_samples)
    ax[1].set_xlabel("Figure 2: Cos signal wave")

    # Automatically adjust subplot spacing
    plt.tight_layout()

    fig.savefig("./figures/ex1.pdf")

