import os
import numpy as np
import matplotlib.pyplot as plt

figures_directory = './figures'


def sawtooth(t: float, f=1, a=1, phi=0) -> float:
    return 2 * a * np.mod(f * t + phi, a) - a


def square(t: float, f=1, a=1, phi=0) -> float:
    return a * np.sign(np.sin(2 * np.pi * f * t + phi))


if __name__ == "__main__":
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory, exist_ok=True)

    fig, ax = plt.subplots(nrows=3, ncols=1)

    nof_samples = 1000
    a = 1
    f = 10
    phi = 2
    samples = np.linspace(0, 1, nof_samples)

    # sawtooth wave plot
    sawtooth_wave = np.array(sawtooth(samples, f=f, a=a))
    ax[0].plot([0, 0], [a, -a])
    ax[0].plot(samples, [0] * nof_samples)
    ax[0].plot(samples, sawtooth_wave)
    ax[0].set_xlabel('Figure 1: Sawtooth wave')

    # square wave plot
    square_wave = np.array(square(samples, f=f, a=a, phi=phi))
    ax[1].plot([0, 0], [a, -a])
    ax[1].plot(samples, [0] * nof_samples)
    ax[1].plot(samples, square_wave)
    ax[1].set_xlabel('Figure 2: Square wave')

    # combined wave plot
    combined_wave = sawtooth_wave + square_wave
    ax[2].plot([0, 0], [max(combined_wave), min(combined_wave)])
    ax[2].plot(samples, [0] * nof_samples)
    ax[2].plot(samples, combined_wave)
    ax[2].set_xlabel('Figure 3: Sawtooth + Square wave')

    fig.suptitle('Wave addition')
    fig.tight_layout()
    fig.savefig(f"./{figures_directory}/ex4.pdf")
