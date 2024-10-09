'''
8. ˆIn practic˘a se opereaz˘a des cu urm˘atoarea aproximare: pentru valori mici
ale lui α, sin(α) ≈ α. Verificat, i dac˘a aceast˘a aproximare este corect˘a,
reprezentˆand grafic cele dou˘a curbe pentru valori ale lui α ˆın intervalul
[−π/2, π/2]. Ar˘atat, i s, i un grafic cu eroarea dintre cele dou˘a funct, ii.
Folosit, i s, i aproximarea Pade sin(α) ≈
α− 7α3
60
1+ α2
20
, nu doar Taylor. Afis,at, i
rezultatele s, i pe un grafic unde axa 0y este logaritmic˘a.
'''

import os
import numpy as np
import matplotlib.pyplot as plt

figures_directory = './figures'


def pade(a: float) -> float:
    return (a - (7/60 * a**3)) / (1 + a**2 / 20)


def sin_approximation() -> None:
    nof_samples = 1000
    samples = np.linspace(-np.pi, np.pi, nof_samples)

    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig.suptitle('x ~ sin(x)')
    ax[0].plot(samples, samples, label='x')
    ax[0].plot(samples, np.sin(samples), label='sin(x)')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].legend()
    ax[0].grid(color='gray', linestyle='dashed')

    approximation_error = np.abs(samples - np.sin(samples))
    ax[1].plot(samples, approximation_error)
    ax[1].grid(color='gray', linestyle='dashed')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('error')
    fig.tight_layout()
    fig.savefig(f"./{figures_directory}/ex8_sin_approximation.pdf")


def pade_approximation() -> None:
    nof_samples = 1000
    samples = np.linspace(-np.pi, np.pi, nof_samples)

    fig, ax = plt.subplots(nrows=3, ncols=1)
    fig.suptitle('Pade approximation')
    ax[0].plot(samples, np.sin(samples), label='sin(x)')
    ax[0].plot(samples, pade(samples), label='pade(x)')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].legend()
    ax[0].grid(color='gray', linestyle='dashed')

    approximation_error = np.abs(np.sin(samples) - np.array(pade(samples)))
    ax[1].plot(samples, approximation_error)
    ax[1].grid(color='gray', linestyle='dashed')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('error')

    ax[2].plot(samples, np.sin(samples), label='sin(x)')
    ax[2].plot(samples, pade(samples), label='pade(x)')
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('log(y)')
    ax[2].legend()
    ax[2].set_yscale('log')
    ax[2].grid(color='gray', linestyle='dashed')

    fig.tight_layout()
    fig.savefig(f"./{figures_directory}/ex8_pade_approximation.pdf")


if __name__ == '__main__':
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory, exist_ok=True)

    sin_approximation()
    pade_approximation()