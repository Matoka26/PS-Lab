import numpy as np
import matplotlib.pyplot as plt
from utils import sinusoidal
import os

figures_dir = './figures'

def autocorelation(v: np.array) -> np.array:
    ret = np.zeros(len(v))
    for k in range(len(v)):
        slice = np.pad(v[len(v)-k-1:], (0, len(v)-k-1), mode='constant')
        ret[k] = np.dot(slice, v)
    return ret


def ar_predicitons(series: np.array, p: int, nof_predictions: int = 10) -> np.array:
    # make equal coefficents
    x = np.ones(p) / p
    predictions = []
    for _ in range(nof_predictions):
        # slice last p rows
        last_vals = series[-p:]
        prediction = np.dot(last_vals, x)
        predictions.append(prediction)
        series = np.append(series, prediction)

    return np.array(predictions)


if __name__ == "__main__":
    if __name__ == "__main__":
        if not os.path.isdir(figures_dir):
            os.makedirs(figures_dir, exist_ok=True)

    N = 1000
    samples = np.arange(0, N)

    trend = [0.00002 * x**2 + 0.0001 * x + 1 for x in samples]
    seassonal = sinusoidal(1, 0.3, samples, 0) + sinusoidal(1, 0.7, samples, 0)
    residuals = np.random.normal(2, 2, size=N)
    observed = trend + seassonal + residuals

    fig,ax = plt.subplots(nrows=4, ncols=1)

    ax[0].plot(samples, trend)
    ax[0].set_title("Trend")

    ax[1].plot(samples, seassonal)
    ax[1].set_title("Seassonal")

    ax[2].plot(samples, residuals)
    ax[2].set_title("Residuals")

    ax[3].plot(samples, observed)
    ax[3].set_title("Observed")

    for x in ax:
        x.grid(True)

    fig.tight_layout()
    plt.savefig(f'{figures_dir}/ex1_a.pdf')

    plt.clf()
    np_corelation = np.correlate(observed, observed, "full")
    np_corelation = np_corelation[:len(np_corelation)//2 + 1]

    my_corelation = autocorelation(observed)

    plt.plot(samples, np_corelation, label="np")
    plt.plot(samples, my_corelation, label="mine")

    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.grid(True)

    # print(f'Difference from my correlation and numpy {np.linalg.norm(np_corelation-my_corelation)}')

    plt.savefig(f'{figures_dir}/ex1_b.pdf')
    plt.clf()

    m = 100
    p = 100
    prediction = ar_predicitons(observed, m, p)
    plt.plot(samples, observed, c='b', label="current")
    plt.plot(np.arange(N, N+p), prediction, c='r', label="predicted")

    plt.grid(True)
    plt.legend()
    plt.savefig(f'{figures_dir}/ex1_c.pdf')