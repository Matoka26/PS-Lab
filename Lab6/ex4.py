'''
 Fis, ierul Train.csv [2] cont, ine date de trafic ˆınregistrate pe o perioad˘a de
1 s˘apt˘amˆan˘a. Perioada de es,antionare este de 1 or˘a, iar valorile m˘asurate
reprezint˘a num˘arul de vehicule ce trec printr-o anumit˘a locat, ie.
(a) Selectat, i din semnalul dat o port, iune corespunz˘atoare pentru 3 zile,
x, pe care vet, i lucra ˆın continuare.
(b) Utilizat, i funct, ia np.convolve(x, np.ones(w), ’valid’) / w pentru a realiza un filtru de tip medie alunec˘atoare s, i netezit, i semnalul
obt, inut anterior. Setat, i dimensiuni diferite ale ferestrei (variabila w
ˆın codul de mai sus), spre exemplu 5, 9, 13, 17.
(c) Dorind s˘a filtrat, i zgomotul (frecvent,e ˆınalte) din semnalul cu date de
trafic, aleget, i o frecvent,˘a de t˘aiere pentru un filtru trece-jos pe care
ˆıl vet, i crea ˆın continuare. Argumentat, i. Care este valoarea frecvent,ei
ˆın Hz s, i care este valoarea frecvent,ei normalizate ˆıntre 0 s, i 1, unde 1
reprezint˘a frecvent,a Nyquist?
(d) Utilizˆand funct, iile s, i scipy.signal.butter s, i scipy.signal.cheby1
proiectat, i filtrele Butterworth s, i Chebyshev de ordin 5, cu frecvent,a
de t˘aiere Wn stabilit˘a mai sus. Pentru ˆınceput setat, i atenuarea
ondulat, iilor, rp = 5 dB, urmˆand ca apoi s˘a ˆıncercat, i s, i alte valori.
(e) Filtrat, i datele de trafic cu cele 2 filtre s, i afis,at, i semnalele filtrate
ˆımpreun˘a cu datele brute. Ce filtru aleget, i din cele 2 s, i de ce?
(f) Reproiectat, i filtrele alegˆand atˆat un ordin mai mic, cˆat s, i unul mai
mare. De asemenea, reproiectat, i filtrul Chebyshev cu alte valori
ale rp s, i observat, i efectul. Stabilit, i val
'''

'''
Answers:
    # c)
    # e)
    # f)
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss


figures_directory = './figures'

if __name__ == "__main__":
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory, exist_ok=True)

    data = pd.read_csv("datasets/Train.csv", parse_dates=["Datetime"], dayfirst=True)
    samples = data["ID"].values[1:]
    wave = data["Count"].values[1:]

    # a)
    nof_hours = 72
    sample_rate = 1 / 3600
    samples = samples[:nof_hours]
    wave = wave[:nof_hours]

    # b)
    ws = [5, 9, 13, 17]
    for w in ws:
        aux = np.convolve(wave, np.ones(w), 'valid') / w
        plt.plot(samples[:len(aux)], aux, label=f'w:{w}')
    plt.legend()
    plt.grid(True)
    plt.title("Window size variation for convolution")
    plt.savefig(f"./{figures_directory}/ex4_b.pdf")
    plt.clf()

    # d)
    filter_order = 5
    Wn = sample_rate * 1/10
    rp = 5
    b_butter, a_butter = ss.butter(filter_order, Wn, analog=False, output="ba", btype="lowpass", fs=sample_rate)
    b_cheby, a_cheby = ss.cheby1(filter_order, rp, Wn, analog=False, output="ba",btype="lowpass", fs=sample_rate)

    filtered_butter = ss.lfilter(b_butter, a_butter, wave)
    filtered_cheby = ss.lfilter(b_cheby, a_cheby, wave)

    plt.plot(samples, wave, label="Actual Signal")
    plt.plot(samples, filtered_butter, label="Butter")
    plt.plot(samples, filtered_cheby, label="Cheby")
    plt.legend()
    plt.grid(True)

    plt.title("Butter & Cheby")
    plt.savefig(f"./{figures_directory}/ex4_d.pdf")
    plt.clf()

    # f)
    plt.plot(samples, wave, label="Actual Signal")
    rps = [1e-6, 1, 7, 30]
    for rp in rps:
        b_cheby, a_cheby = ss.cheby1(filter_order, rp, Wn, analog=False, output="ba", btype="lowpass", fs=sample_rate)
        filtered_cheby = ss.lfilter(b_cheby, a_cheby, wave)
        plt.plot(samples, filtered_cheby, label=f'rp:{rp}')
    plt.legend()
    plt.grid(True)
    plt.title("Rp variation for Cheby")
    plt.savefig(f"./{figures_directory}/ex4_f1.pdf")
    plt.clf()

    # order variation
    plt.plot(samples, wave, label="Actual Signal")
    filter_order = [1, 5, 9]
    rp = 5
    for filter in filter_order:
        b_cheby, a_cheby = ss.cheby1(filter, rp, Wn, analog=False, output="ba", btype="lowpass", fs=sample_rate)
        filtered_cheby = ss.lfilter(b_cheby, a_cheby, wave)
        plt.plot(samples, filtered_cheby, label=f'filter order:{filter}')
    plt.legend()
    plt.grid(True)
    plt.title("Filter order variation for Cheby")
    plt.savefig(f"./{figures_directory}/ex4_f2.pdf")
    plt.clf()
