'''
Fie semnalele continue x(t) = cos(520πt + π/3), y(t) = cos(280πt − π/3)
s, i z(t) = cos(120πt + π/3).
    (a) ˆIn Python, simulat, i axa real˘a de timp printr-un s, ir de numere suficient de apropiate, spre exemplu [0 : 0.0005 : 0.03].

    (b) Construit, i semnalele x(t), y(t) s, i z(t) s, i afis,at, i-le grafic, ˆın cate un
    subplot.

    (c) Es,antionat, i semnalele cu o frecvent,˘a de 200 Hz pentru a obt, ine x[n],
    y[n] s, i z[n] s, i afis,at, i-le grafic, ˆın cˆate un subplot.
'''

import numpy as np
import matplotlib.pyplot as plt

no_samples = 500
range_start = 0
range_step = 0.0005
range_stop = 0.03


def x(t: float, f=520) -> float:
    return np.cos(np.pi * t * f + np.pi / 3)


def y(t: float, f=280) -> float:
    return np.cos(np.pi * t * f - np.pi / 3)


def z(t: float, f=120) -> float:
    return np.cos(np.pi * t * f + np.pi / 3)


# a)
samples = np.arange(range_start, range_stop, range_step)

# b)
x_axis = x(samples)
y_axis = y(samples)
z_axis = z(samples)

fig, ax = plt.subplots(nrows=3, ncols=1)
ax[0].plot(samples, x_axis, c='r')
ax[1].plot(samples, y_axis, c='b')
ax[2].plot(samples, z_axis, c='g')

fig.savefig("./figures/ex1_b.pdf")

# c) ?

x_axis = x(samples, 200)
y_axis = y(samples, 200)
z_axis = z(samples, 200)

fig, ax = plt.subplots(nrows=3, ncols=1)
ax[0].plot(samples, x_axis, c='r')
ax[1].plot(samples, y_axis, c='b')
ax[2].plot(samples, z_axis, c='g')

fig.savefig("./figures/ex1_c.pdf")