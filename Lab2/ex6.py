'''
6. Generat, i 3 semnale de tip sinus cu amplitudine unitar˘a s, i faz˘a nul˘a avˆand
frecvent,ele fundamentale:
(a) f = fs/2
(b) f = fs/4
(c) f = 0 Hz
unde fs este frecvent,a de es,antionare, aleas˘a de voi. Notat, i ce observat, i.
'''
import os
import numpy as np
import matplotlib.pyplot as plt

figures_directory = './figures'


def sinusoidal(a: float, f: int, t: float, phi: float) -> float:
    return a * np.sin(2 * np.pi * f * t + phi)


if __name__ == '__main__':
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory, exist_ok=True)

    nof_samples = 400
    fs = nof_samples
    a = 1
    phi = 0

    samples = np.linspace(0, 1, nof_samples)
    fig, ax = plt.subplots(nrows=3, ncols=1)

    # a)
    ax[0].plot([0, 0], [a, -a])
    ax[0].plot(samples, [0] * nof_samples)
    ax[0].plot(samples, sinusoidal(a, fs/2, samples, phi))
    ax[0].set_xlabel(f'Figure 1: f = {int(fs/2)}, fs = {nof_samples}')

    # b)
    ax[1].plot([0, 0], [a, -a])
    ax[1].plot(samples, [0] * nof_samples)
    ax[1].plot(samples, sinusoidal(a, fs/4, samples, phi))
    ax[1].set_xlabel(f'Figure 2: f = {int(fs/4)}, fs = {nof_samples}')

    # c)
    ax[2].plot([0, 0], [a, -a])
    ax[2].plot(samples, [0] * nof_samples)
    ax[2].plot(samples, sinusoidal(a, 0, samples, phi))
    ax[2].set_xlabel(f'Figure 3: f = 0 Hz,  fs = {nof_samples}')

    fig.tight_layout()
    fig.suptitle('Fundamental frequency variation')
    fig.savefig(f"./{figures_directory}/ex6.pdf")

    '''
    NOTE: Frecventa de esantionare este mult prea mica iar semnalele sunt deformate.
        Conform Teoremei Nyquist-Shannon frecventa de esantionare trebuie sa fie 
        'mult mai mare' deact dublul frecventei fundamentale
        
        Teorema Nyquist-Shannon:
        https://epe.utcluj.ro/SCTR/Indicatii/Teorema_lui_Nyquist_Shannon.pdf
    '''