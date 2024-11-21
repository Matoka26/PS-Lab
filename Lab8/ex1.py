import numpy as np
import matplotlib.pyplot as plt
from utils import sinusoidal


def autocorelation(v: np.array) -> np.array:
    ret = np.zeros(len(v))
    for k in range(len(v)):
        slice = np.pad(v[len(v)-k-1:], (len(v)-k-1, 0), mode='constant')
        ret[k] = np.dot(slice, v) / np.std(v) * np.std(slice)
    return ret


if __name__ == "__main__":
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
    # plt.show()

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

    print(np.linalg.norm(np_corelation-my_corelation))

    plt.show()