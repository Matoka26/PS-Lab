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
    return 2 * a * np.mod(f * t + phi, a) - a


def square(t: float, f=1, a=1, phi=0) -> float:
    return a * np.sign(np.sin(2 * np.pi * f * t + phi))


def s(x: int, y: int, u: float, v: float) -> float:
    return np.sin(2 * np.pi * (x * u + y * v))


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
    f = 240
    a = 1
    phi = 0
    samples = np.linspace(0, 1, no_samples)
    plt.plot(samples, sawtooth(samples, f, a, phi), c='r')
    plt.plot([0,0], [a, -a])
    plt.plot(samples, [0]*no_samples)
    plt.savefig("./figures/ex2_c.pdf")


# d)
def ex_d():
    no_samples = 5000
    f = 300
    a = 1
    phi = 0
    samples = np.linspace(0, 1, no_samples)
    plt.plot(samples, square(samples, f, a, phi), c='r')
    plt.plot([0,0], [a, -a])
    plt.plot(samples, [0]*no_samples)
    plt.savefig("./figures/ex2_d.pdf")


# e)
def ex_e():
    img = np.random.rand(128, 128)
    plt.imshow(img, cmap='gray')
    plt.savefig("./figures/ex2_e.pdf")


# f)
def ex_f():
    u = 2
    v = -3
    img = np.zeros((128, 128))
    for x in range(len(img)):
        for y in range(len(img[0])):
            img[x][y] = s(x, y, u, v)
    plt.imshow(img, cmap='grey')
    plt.savefig("./figures/ex2_f.pdf")


if __name__ == "__main__":
   ex_a()
   ex_b()
   ex_c()
   ex_d()
   ex_e()
   ex_f()