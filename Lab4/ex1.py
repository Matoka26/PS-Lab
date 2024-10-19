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
json_file_path = './temp.out'



if __name__ == '__main__':
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory, exist_ok=True)

    N = [2 ** i for i in range(7, 10)]

    df_times = dict()
    df_times['my_dft_times'] = np.zeros(len(N))
    df_times['np_fft_times'] = np.zeros(len(N))

    # plt.plot(samples, wave)
    # plt.show()

    for i, n in enumerate(N):
        samples, wave = get_random_signal(5, n)

        # time np_fft.fft
        start_time = time.time()
        np.fft.fft(wave)
        df_times['my_dft_times'][i] = time.time() - start_time

        # time my dft
        start_time = time.time()
        get_fourier_components(wave, n)
        df_times['np_fft_times'][i] = time.time() - start_time

    with open(json_file_path, 'w') as json_file:
        json_string = json_file.dumps(df_times)

        print(json_string)