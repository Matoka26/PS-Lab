'''
Demonstrat, i (grafic) c˘a alegˆand o frecvent,˘a de es,antionare mai mare decˆat
frecvent,a Nyquist, nu mai obt, inet, i fenomenul de aliere pentru semnalul
ales la exercit, iul precedent. La fel ca mai sus, indicat, i es,antioanele s, i
pentru celelalte dou˘a semnale construite.
'''

from utils import sinusoidal
import matplotlib.pyplot as plt
import numpy as np
import os

figures_directory1 = './figures_pdf'
figures_directory2 = './figures_png'

if __name__ == '__main__':
    if not os.path.isdir(figures_directory1):
        os.makedirs(figures_directory1, exist_ok=True)
    if not os.path.isdir(figures_directory2):
        os.makedirs(figures_directory2, exist_ok=True)

    f = 100
    a = 1
    phi = 0
    nofs_samples = [f*2, f*3, f*4, f*5]

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
    fig.savefig(f"./{figures_directory1}/ex3.pdf")
    fig.savefig(f"./{figures_directory2}/ex3.png")