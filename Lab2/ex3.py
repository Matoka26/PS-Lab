'''
Generat, i urm˘atoarele semnale s, i afis,at, i-le grafic, fiecare ˆıntr-un plot:
    (a) Un semnal sinusoidal de frecvent,˘a 400 Hz, care s˘a cont, in˘a 1600 de
    es,antioane.

    (b) Un semnal sinusoidal de frecvent,˘a 800 Hz, care s˘a dureze 3 secunde.

    (c) Un semnal de tip sawtooth de frecvent,˘a 240 Hz (putet, i folosi funct, iile
    numpy.floor sau numpy.mod).

    (d) Un semnal de tip square de frecvent,˘a 300 Hz (putet, i folosi funct, ia
    numpy.sign).

    (e) Un semnal 2D aleator. Creat, i un numpy.array de dimensiune 128x128
    s, i init, ializat, i-l aleator, folosind numpy.random.rand(x,y), unde x s, i
    y reprezint˘a num˘arul de linii respectiv de coloane. Afis,at, i semnalul
    generat folosind funct, ia imshow(I) din matplotlib.

    (f) Un semnal 2D la alegerea voastr˘a. Creat, i un numpy.array de dimensiune 128x128 s, i init, ializat, i-l folosind o procedur˘a creat˘a de voi.
    Utilizat, i, spre exemplu, funct, iile numpy.zeros() s, i numpy.ones().
'''
import os
import numpy as np
import sounddevice as sd
from scipy.io import wavfile

wave_sounds_directory = 'wave_sounds'


def play_wave(wave: np.array, file_name: str, audio_sample_rate: int = 44100) -> None:
    if not os.path.isdir(wave_sounds_directory):
        os.makedirs(wave_sounds_directory, exist_ok=True)

    wavfile.write(f'./{wave_sounds_directory}/{file_name}.wav', audio_sample_rate, wave)
    sd.play(wave, samplerate=audio_sample_rate, loop=False)     # Make loop=True to actually hear it, it ends too quick
    sd.wait()


def sinusoidal(t: float, f=1, a=1, phi=0) -> float:
    return a * np.sin(2 * np.pi * f * t + phi)


def sawtooth(t: float, f=1, a=1, phi=0) -> float:
    return 2 * a * np.mod(f * t + phi, a) - a


def square(t: float, f=1, a=1, phi=0) -> float:
    return a * np.sign(np.sin(2 * np.pi * f * t + phi))


def s(x: int, y: int, u: float, v: float) -> float:
    return np.sin(2 * np.pi * (x * u + y * v))


# a)
def ex_a():
    nof_samples = 1600
    f = 400
    a = 1
    phi = 0
    samples = np.linspace(0, 1, nof_samples)
    wave = sinusoidal(samples, f, a, phi)

    play_wave(wave, "lab1_ex2_a")


# b)
def ex_b():
    nof_samples = 1600
    f = 800
    a = 1
    phi = 0
    samples = np.linspace(0, 3, nof_samples)
    wave = sinusoidal(samples, f, a, phi)

    play_wave(wave, "lab1_ex2_b")


# c)
def ex_c():
    nof_samples = 5000
    f = 240
    a = 1
    phi = 0
    samples = np.linspace(0, 1, nof_samples)
    wave = sawtooth(samples, f, a, phi)

    play_wave(wave, "lab1_ex2_c")


# d)
def ex_d():
    nof_samples = 5000
    f = 300
    a = 1
    phi = 0
    samples = np.linspace(0, 1, nof_samples)
    wave = square(samples, f, a, phi)

    play_wave(wave, "lab1_ex2_d")


if __name__ == "__main__":
    ex_a()
    ex_b()
    ex_c()
    ex_d()

    # test existing .wav file
    sample_rate, audio_data = wavfile.read(f'./{wave_sounds_directory}/lab1_ex2_a.wav')
    sd.play(audio_data, samplerate=sample_rate, loop=True)
    sd.wait()