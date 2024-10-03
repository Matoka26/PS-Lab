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

import numpy as np
import matplotlib.pyplot as plt


def sinusoidal(t: float, f=1, a=1, phi=0) -> float:
    return a * np.sin(2 * np.pi * f * t + phi)


def sawtooth(t: float, f=1, a=1, phi=0) -> float:
    return a * np.abs(2 * np.pi * f * t + phi) % a


# a)
def ex_a():

    no_samples = 1600
    f = 400
    a = 1
    phi = 0
    samples = np.linspace(0, 1, no_samples)
    plt.plot(sinusoidal(samples, f, a, phi), c='r')

    plt.plot([0,0], [a, -a])
    plt.plot(samples, [0]*no_samples)
    plt.savefig("./figures/ex2_a.pdf")


# b)
def ex_b():
    no_samples = 1600
    f = 800
    a = 1
    phi = 0
    samples = np.linspace(0, 3, no_samples)
    plt.plot(samples, sinusoidal(samples, f, a, phi), c='r')
    plt.plot([0,0], [a, -a])
    plt.plot(samples, [0]*no_samples)
    plt.savefig("./figures/ex2_b.pdf")


# c)
def ex_c():
    no_samples = 5000
    f = 100
    a = 1
    phi = 0
    samples = np.linspace(0, 1, no_samples)
    plt.plot(samples, sawtooth(samples, f, a, phi), c='r')
    plt.plot([0,0], [a, -a])
    plt.plot(samples, [0]*no_samples)
    plt.show()
    # plt.savefig("./figures/ex2_c.pdf")

if __name__ == "__main__":
   # ex_a()
   # ex_b()
    ex_c()