import os
import numpy as np
import matplotlib.pyplot as plt

figures_directory = './figures'


def sinusoidal(a: float, f: int, t: float, phi: float) -> float:
    return a * np.sin(2 * np.pi * f * t + phi)


def get_complex_representation(x: np.array, omega=1) -> np.array:
    return [x[i] * np.exp(-2j * np.pi * i * omega / len(x)) for i in range(len(x))]


def plot_fig_1(samples: np.array, x: np.array, y: np.array, a: float) -> None:
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].plot(samples, x)
    ax[0].plot([0, 0], [a, -a])
    ax[0].plot(samples, [0] * len(y))
    ax[0].set_xlabel('Samples')
    ax[0].set_ylabel('Amplitute')

    ax[1].plot([samp.real for samp in y], [samp.imag for samp in y])
    ax[1].plot([a, -a], [0, 0])
    ax[1].plot([0, 0], [a, -a])
    ax[1].set_xlabel('Real')
    ax[1].set_ylabel('Imaginary')
    ax[1].grid(True)

    fig.tight_layout()
    fig.savefig(f"./{figures_directory}/ex2_fig_1.pdf")


def plot_fig_2(x: np.array, a: float) -> None:
    fig, ax = plt.subplots(nrows=2, ncols=2)

    omegas = [2, 3, 4, 5]
    for i,ome in enumerate(omegas):
        z = get_complex_representation(x, ome)
        ax[i%2, int(i < 2)].plot([samp.real for samp in z], [samp.imag for samp in z])
        ax[i%2, int(i < 2)].plot([a, -a], [0, 0])
        ax[i%2, int(i < 2)].plot([0, 0], [a, -a])
        ax[i%2, int(i < 2)].set_xlabel('Real')
        ax[i%2, int(i < 2)].set_ylabel('Imaginary')
        ax[i%2, int(i < 2)].grid(True)

    fig.tight_layout()
    fig.savefig(f"./{figures_directory}/ex2_fig_2.pdf")


if __name__ == '__main__':
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory, exist_ok=True)

    nof_samples = 1000
    f = 7
    a = 3.4
    phi = np.pi / 2
    samples = np.linspace(0, 1, nof_samples)
    x = sinusoidal(a, f, samples, phi)
    y = get_complex_representation(x)

    plot_fig_1(samples, x, y, a)
    plot_fig_2(x, a)

