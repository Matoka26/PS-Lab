'''
Generat, i un vector x[n] aleator de dimensiune N = 100. Calculat, i iterat, ia
x ← x ∗ x de trei ori. Afis,at, i cele patru grafice. Ce observat, i?
'''

import os
import numpy as np
import matplotlib.pyplot as plt

figures_directory = './figures'

if __name__ == "__main__":
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory, exist_ok=True)

    n = 100
    nof_iterations = 4
    wave = np.random.rand(n)
    samples = np.linspace(0, 1, n)

    x = wave
    plt.plot(samples, x)
    for i in range(nof_iterations-1):
        x = np.convolve(x, wave)
        samples = np.linspace(0, 1, len(x))
        plt.plot(samples, x)

    plt.grid(True)
    plt.title(f'{nof_iterations} Convolutions')
    plt.savefig(f"./{figures_directory}/ex1.pdf")