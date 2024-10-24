'''
 Construit, i un semnal sinusoidal de frecvent,˘a aleas˘a de voi, de amplitudine unitar˘a s, i faz˘a nul˘a. Demonstrat, i (grafic) c˘a es,antionarea lui cu o
frecvent,˘a sub-Nyquist (aleas˘a, de asemenea, de voi) genereaz˘a fenomenul
de aliere. Pentru aceasta creat, i alte dou˘a semnale, de frecvent,e diferite,
care es,antionate cu frecvent,a aleas˘a mai sus produc aceleas, i es,antioane ca
semnalul init, ial. Obt, inet, i, astfel, o figur˘a similar˘a Figurii 2.

'''

from utils import sinusoidal
import matplotlib.pyplot as plt
import numpy as np
import os

figures_directory1 = './figures_png'
figures_directory2 = './figures_pdf'

if __name__ == '__main__':
    if not os.path.isdir(figures_directory1):
        os.makedirs(figures_directory1, exist_ok=True)
    if not os.path.isdir(figures_directory2):
        os.makedirs(figures_directory2, exist_ok=True)

    f = 100
    a = 1
    phi = 0
    nofs_samples = [int(f/2), f, int(f*1.5), f*2]

    fig, ax = plt.subplots(ncols=1, nrows=len(nofs_samples))

    for i, nof_samples in enumerate(nofs_samples):
        samples = np.linspace(0, 1, nof_samples)
        wave = sinusoidal(a, f, samples, phi)

        ax[i].grid()
        ax[i].plot(samples, wave)
        ax[i].plot([0, 0], [a, -a])
        ax[i].plot(samples, [0] * nof_samples)
        ax[i].set_xlabel(f"Figure {i+1}: Sample rate {1/nof_samples}s")

    fig.tight_layout()
    fig.savefig(f"./{figures_directory2}/ex2.pdf")
    fig.savefig(f"./{figures_directory1}/ex2.png")