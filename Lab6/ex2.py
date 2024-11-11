'''
Vi se dau dou˘a polinoame p(x) s, i q(x) cu grad maxim N generate aleator
cu coeficient, i ˆıntregi. Calculat, i produsul lor r(x) = p(x)q(x) folosing
convolut, ia: folosind ˆınmult, irea polinoamelor direct˘a s, i apoi folosind fft.
'''

import numpy as np

if __name__ == "__main__":

    N = 5
    p_coef = np.random.rand(N)
    q_coef = np.random.rand(N)

    # Direct multiplication
    r_coef_direct = np.convolve(p_coef, q_coef)
    print(f'Direct convolution: {r_coef_direct}')

    # FFT Method
    p_fft = np.fft.fft(p_coef, 2 * N - 1)
    q_fft = np.fft.fft(q_coef, 2 * N - 1)
    r_fft = np.multiply(p_fft, q_fft)
    r_coef_fft = np.real(np.fft.ifft(r_fft))
    print(f'FFT method: {r_coef_fft}')

    print(f'Are the same? {np.allclose(r_coef_fft, r_coef_direct)}')
