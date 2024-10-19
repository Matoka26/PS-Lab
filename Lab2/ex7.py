'''
7. Generat, i un semnal sinusoidal cu frecvent,a de es,antionare 1000 Hz s, i
decimat, i-l la 1/4 din frecvent,a init, ial˘a (p˘astrat, i doar al 4-lea fiecare element din vector):
(a) Afis,at, i grafic cele dou˘a semnale s, i comentat, i diferent,ele.
(b) Repetat, i decimarea (tot la 1/4 din frecvent,a init, ial˘a) pornind acum
de la al doilea element din vector. Ce observat, i?

'''
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import sinusoidal

figures_directory = './figures'


if __name__ == '__main__':
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory, exist_ok=True)

    nof_samples = 1000
    a = 2
    f = 150
    phi = np.e

    samples = np.linspace(0, 1, nof_samples)
    fig, ax = plt.subplots(nrows=3, ncols=1)

    ax[0].plot([0, 0], [a, -a])
    ax[0].plot(samples, [0] * nof_samples)
    ax[0].plot(samples, sinusoidal(a, f, samples, phi))
    ax[0].set_xlabel('Figure 1: Initial frequency')

    first_quarter = np.array([x for i, x in enumerate(samples) if i % 4 == 0])
    ax[1].plot([0, 0], [a, -a])
    ax[1].plot(first_quarter, [0] * first_quarter)
    ax[1].plot(first_quarter, sinusoidal(a, f, first_quarter, phi))
    ax[1].set_xlabel('Figure 2: Decimated from first position')

    second_quarter = np.array([x for i, x in enumerate(samples) if i % 4 == 1])
    ax[2].plot([0, 0], [a, -a])
    ax[2].plot(second_quarter, [0] * second_quarter)
    ax[2].plot(second_quarter, sinusoidal(a, f, second_quarter, phi))
    ax[2].set_xlabel('Figure 3: Decimated from second position')

    fig.tight_layout()
    fig.suptitle('Decimate sampling frequency')
    fig.savefig(f"./{figures_directory}/ex7.pdf")

    '''
    NOTE: Pentru frecvente mari, semnalele decimate arata ca versiuni de faze
        diferite ale aceluiasi semnal
    '''
