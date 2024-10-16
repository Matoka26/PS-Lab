import os
import numpy as np
import matplotlib.pyplot as plt

figures_directory1 = './figures_pdf'
figures_directory2 = './figures_png'

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True


def sinusoidal(a: float, f: int, t: float, phi: float) -> float:
    return a * np.sin(2 * np.pi * f * t + phi)


def get_complex_representation(x: np.array, omega=1) -> np.array:
    return [x[i] * np.exp(-2j * np.pi * i * omega / len(x)) for i in range(len(x))]


def plot_fig_1(samples: np.array, x: np.array, y: np.array, a: float) -> None:
    distances = np.abs(y)
    # Normalize distances for color mapping
    norm = plt.Normalize(distances.min(), distances.max())
    colors = plt.cm.viridis(norm(distances))

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].scatter(samples, x, c=colors, s=5)
    ax[0].plot([0, 0], [a, -a])
    ax[0].plot(samples, [0] * len(y))
    ax[0].set_xlabel('Samples')
    ax[0].set_ylabel('Amplitute')

    ax[1].scatter([samp.real for samp in y], [samp.imag for samp in y], c=colors, s=5)
    ax[1].plot([a, -a], [0, 0])
    ax[1].plot([0, 0], [a, -a])
    ax[1].set_xlabel('Real')
    ax[1].set_ylabel('Imaginary')
    ax[1].grid(True)

    fig.tight_layout()
    fig.savefig(f"./{figures_directory1}/ex2_fig_1.pdf")
    fig.savefig(f"./{figures_directory2}/ex2_fig_1.png")


def plot_fig_2(x: np.array, a: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 8),nrows=2, ncols=2)

    omegas = [2, 3, 4, 5]
    for i, ome in enumerate(omegas):
        z = get_complex_representation(x, ome)

        distances = np.abs(z)
        # Normalize distances for color mapping
        norm = plt.Normalize(distances.min(), distances.max())
        colors = plt.cm.viridis(norm(distances))

        scatter = ax[i % 2, int(i < 2)].scatter([samp.real for samp in z], [samp.imag for samp in z], c=colors, s=5)

        ax[i % 2, int(i < 2)].plot([a, -a], [0, 0], color='black')
        ax[i % 2, int(i < 2)].plot([0, 0], [a, -a], color='black')
        ax[i % 2, int(i < 2)].set_xlabel('Real')
        ax[i % 2, int(i < 2)].set_ylabel('Imaginary')
        ax[i % 2, int(i < 2)].grid(True)
        ax[i % 2, int(i < 2)].set_title(f'omega={ome}')

    fig.tight_layout()
    fig.savefig(f"./{figures_directory1}/ex2_fig_2.pdf")
    fig.savefig(f"./{figures_directory2}/ex2_fig_2.png")


if __name__ == '__main__':
    if not os.path.isdir(figures_directory1):
        os.makedirs(figures_directory1, exist_ok=True)

    if not os.path.isdir(figures_directory2):
        os.makedirs(figures_directory2, exist_ok=True)

    nof_samples = 1000
    f = 2
    a = 3.4
    phi = np.pi / 2
    samples = np.linspace(0, 1, nof_samples)
    x = sinusoidal(a, f, samples, phi)
    y = get_complex_representation(x)

    plot_fig_1(samples, x, y, a)
    plot_fig_2(x, a)

