'''
Scriet, i cˆate o funct, ie prin care s˘a construit, i o fereastr˘a dreptunghiular˘a s, i
o fereastr˘a de tip Hanning. Funct, iile primesc ca parametru dimensiunea
ferestrei. Afis,at, i grafic o sinusoid˘a cu f = 100, A = 1 s, i φ = 0 trecut˘a
prin cele dou˘a tipuri de ferestre de dimensiune Nw = 200.
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from utils import sinusoidal

figures_directory = './figures'


def sliding_filter(arr: np.array, window_size: int, function) -> np.ndarray:
    arr = np.concatenate((arr, np.zeros(window_size)))
    res = np.zeros(len(arr))
    window = deque([0] * (window_size-1))

    for i, num in enumerate(arr):
        window.append(function(i) * num)
        res[i] = np.average(window)
        window.popleft()

    return res


if __name__ == "__main__":
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory, exist_ok=True)

    nof_samples = 10000
    f = 100
    a = 1
    phi = 0
    N = 200

    functions = {
        'rectangle': lambda x: 1,
        'hanning': lambda x: 0.5 * (1 - np.cos(2 * np.pi * x/N)),
    }

    samples = np.linspace(0, 1, nof_samples)
    wave = sinusoidal(a, f, samples, phi)

    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].plot(samples, wave)
    ax[0].set_title("Initial wave")
    ax[0].grid(True)

    for i, fun in enumerate(functions):
        res = sliding_filter(wave, N, functions[fun])
        samples = np.linspace(0, 1, len(res))
        ax[i+1].plot(samples, res)
        ax[i+1].set_title(fun)
        ax[i+1].grid(True)

    fig.tight_layout()
    fig.savefig(f"./{figures_directory}/ex3.pdf")


