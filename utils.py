from numpy import \
    sin, cos, \
    pi, \
    mod, sign, exp, \
    zeros, linspace, \
    random, \
    array, ndarray, \
    arange, \
    random



# Formula for a sin wave with sin function
def sinusoidal(a: float, f: int, t: float, phi: float) -> float:
    return a * sin(2 * pi * f * t + phi)


# Formula for a sin wave with cos function
def cosinusoidal(a: float, f: int, t: float, phi: float) -> float:
    return a * cos(2 * pi * f * t + phi - pi/2)


# Formula for 2D wave
def sinusoidal_2d(x: int, y: int, u: float, v: float) -> float:
    return sin(2 * pi * (x * u + y * v))


# Formula for sawtooth like wave
def sawtooth(t: float, f=1, a=1, phi=0) -> float:
    return 2 * a * mod(f * t + phi, a) - a


# Formula for a square like wave
def square(t: float, f=1, a=1, phi=0) -> float:
    return a * sign(sin(2 * pi * f * t + phi))


# Converts a wave to it's complex representation
def get_complex_representation(x: array, omega=1) -> array:
    return [x[i] * exp(-2j * pi * i * omega / len(x)) for i in range(len(x))]


# Calculates the Fourier matrix for a given size
def get_fourier_matrix(n: int) -> ndarray:
    F = zeros((n, n), dtype=complex)

    for k in range(n):
        for m in range(n):
            F[k, m] = exp(-2j * pi * k * m / n)
    return F


# Calculates a number of components for the Discrete Fourier Transform
# over a given array
def get_fourier_components(x: array, nof_comp: int) -> array:
    N = len(x)
    X = zeros(nof_comp, dtype=complex)

    for m in range(nof_comp):
        sum = 0
        for k in range(nof_comp):
            sum += x[k] * exp(-2j * pi * k * m / N)
        X[m] = sum
    return X


# Caclulates the infers transform of a wave for a given number of components
def get_inverse_fourier_transform(X: array, n: int) -> array:
    x = zeros(n, dtype=complex)

    for m in range(n):
        for k in range(n):
            x[m] += X[k] * exp(2j * pi * k * m / n)
    x /= n
    return x


# Generates a random combination of sin signals for a given number of
# waves and number of samples points
def get_random_signal(nof_sum_signals: int, nof_samples: int) -> array:
    # generate random signal
    frequencies = random.randint(size=nof_sum_signals, low=1, high=100, dtype=int)
    amplitudes = random.randint(size=nof_sum_signals, low=1, high=100, dtype=int)
    phases = zeros(nof_sum_signals)

    samples = linspace(0, 1, nof_samples)

    # sum up signal components
    wave = sinusoidal(amplitudes[0], frequencies[0], samples, phases[0])
    for i in range(1, nof_sum_signals):
        wave += sinusoidal(amplitudes[i], frequencies[i], samples, phases[i])

    return samples, wave


def generate_time_serie(N: int) -> array:
    samples = arange(0, N)
    trend = [0.00002 * x**2 + 0.0001 * x + 1 for x in samples]
    seassonal = sinusoidal(1, 0.3, samples, 0) + sinusoidal(1, 0.7, samples, 0)
    residuals = random.normal(0.5, 0.5, size=N)
    observed = trend + seassonal + residuals

    return observed