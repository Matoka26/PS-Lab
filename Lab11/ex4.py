import numpy as np
from utils import hankel_from_serie, generate_time_serie, hankel_from_matrix


def singular_spectrum_analysis(series: np.ndarray, L: int) -> np.ndarray:
    hankelized_series = hankel_from_serie(series, L)
    x_hats = []

    U, S, V = np.linalg.svd(hankelized_series)

    sum = np.zeros(shape=(U.shape[0], V.shape[1]))
    for i in range(np.min(S.shape)):
        Xi = S[i] * np.outer(U[i], V[i])

        Xi = hankel_from_matrix(Xi)
        sum += Xi
        x_hats.append(Xi)

    print(sum)
    print(hankelized_series)

    return np.array(x_hats)


if __name__ == "__main__":
    N = 1000
    samples = np.arange(0, N)
    series = generate_time_serie(N)

    L = 20

    singular_spectrum_analysis(series, L)