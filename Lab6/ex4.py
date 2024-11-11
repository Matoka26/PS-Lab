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
    
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


figures_directory = './figures'

if __name__ == "__main__":
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory, exist_ok=True)

    data = pd.read_csv("datasets/Train.csv", parse_dates=["Datetime"], dayfirst=True)
    samples = data["ID"].values[1:]
    wave = data["Count"].values[1:]

    # a)
    nof_hours = 72
    samples = samples[:nof_hours]
    wave = wave[:nof_hours]

    # b)
    ws = [5, 9, 13, 17]
    for w in ws:
        aux = np.convolve(wave, np.ones(w), 'valid') / w
        plt.plot(samples[:len(aux)], aux, label=f'w:{w}')
    plt.legend()
    plt.grid(True)
    plt.show()

    # c)