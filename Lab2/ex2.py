import os
import numpy as np
import matplotlib.pyplot as plt

figures_directory = './figures'


def sinusoidal(a: float, f: int, t: float, phi: float) -> float:
    return a * np.sin(2 * np.pi * f * t + phi)


def get_gamma(norm_x: float, norm_z: float, snr: float) -> float:
    return np.sqrt((norm_z ** 2) / (norm_x ** 2) * snr)


if __name__ == "__main__":
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory, exist_ok=True)

    nof_samples = 1000
    f = 3
    a = 1
    phi_values = [0, np.pi / 3, np.pi / 4, 3 * np.pi / 2]
    samples = np.linspace(0, 1, nof_samples)

    # plot axis
    plt.plot([0, 0], [a, -a])
    plt.plot(samples, [0] * nof_samples)

    # plot first 3 signals regularly
    for phi in phi_values:
        plt.plot(samples, sinusoidal(a, f, samples, phi), label=f'phi={phi}')

    plt.legend()
    plt.title("Multiple signals with different phases")
    plt.savefig(f'./{figures_directory}/ex2_phi.pdf')

    snr_values = [0.1, 1, 10, 100]
    chosen_phi = phi_values[0]
    z = np.random.normal(loc=0, scale=1, size=nof_samples)
    norm_x = np.linalg.norm(sinusoidal(a, f, samples, chosen_phi))
    norm_z = np.linalg.norm(z)

    fig, ax = plt.subplots(nrows=len(snr_values), ncols=1)
    for i, snr in enumerate(snr_values):
        snr_signal = sinusoidal(a, f, samples, chosen_phi) + get_gamma(norm_x, norm_z, snr) * z
        ax[i].plot(samples, snr_signal)
        ax[i].set_xlabel(f'Figure {i}: Phi:{chosen_phi}, SNR:{snr}')

    fig.tight_layout()
    fig.suptitle('SNR behaviour')
    fig.savefig(f"./{figures_directory}/ex2_snr.pdf")

