'''
 La laboratorul precedent at, i implementat voi Transformata Fourier Discret˘a. Comparat, i timpul de execut, ie al implementarii voastre cu numpy.fft.
Desenat, i un grafic cu timpii de execut, ie pentru dimensiunile vectorilor
N ∈ {128, 256, 512, 1024, 2048, 4096, 8192}. Folosit, i biblioteca time pentru a calcula timpul de rulare iar la plot pentru axa Oy afis,at, i ambii timpi
pe scar˘a logaritmic˘a.

'''

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from utils import sinusoidal, get_fourier_components, get_random_signal

figures_directory = './figures'
temp_file = './temp_ex1.json'


# Generate the times of the routines and save to a temporary file
def generate_statistics(a, b) -> dict:
    N = [2 ** i for i in range(a, b)]

    df_times = dict()
    df_times['my_dft_times'] = np.zeros(len(N))
    df_times['np_fft_times'] = np.zeros(len(N))

    # Calculate MY_DF and NP_FFT times
    for i, n in enumerate(N):
        samples, wave = get_random_signal(5, n)

        # time np_fft.fft
        start_time = time.time()
        np.fft.fft(wave)
        df_times['np_fft_times'][i] = time.time() - start_time

        # time my dft
        start_time = time.time()
        get_fourier_components(wave, n)
        df_times['my_dft_times'][i] = time.time() - start_time

    # Convert NumPy arrays to lists
    data_serializable = {key: value.tolist() for key, value in df_times.items()}

    with open(temp_file, 'w') as json_file:
        json.dump(data_serializable, json_file, indent=4)

    return df_times


# If the statistics file exist, extract data from it
def get_statistics_from_temp() -> dict:
    with open(temp_file, 'r') as json_file:
        data = json.load(json_file)
    data_with_arrays = {key: np.array(value) for key, value in data.items()}
    return data_with_arrays


if __name__ == '__main__':
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory, exist_ok=True)

    exponent_left_limit = 7
    exponent_right_limit = 14

    # Get or generate statistics
    if os.path.isfile(temp_file):
        statistics = get_statistics_from_temp()
        if len(statistics[next(iter(statistics))]) != exponent_right_limit - exponent_left_limit:
            statistics = generate_statistics(exponent_left_limit, exponent_right_limit)
    else:
        statistics = generate_statistics(exponent_left_limit, exponent_right_limit)

    eps = 1e-5

    scales = {
        0: 'linear',
        1: 'log'
    }
    fig, ax = plt.subplots(ncols=1, nrows=len(scales))

    for key, data in statistics.items():
        n = len(data)

        for i in scales.keys():
            ax[i].set_yscale(scales[i])
            ax[i].plot([i+eps for i in range(n)], data, label=key)

    for i in scales.keys():
        ax[i].set_ylabel(f'Times - {scales[i]}(s)')
        ax[i].set_xlabel('Nof Components(Exponential)')
        ax[i].grid()

    fig.tight_layout()
    fig.savefig(f"./{figures_directory}/ex1.pdf")