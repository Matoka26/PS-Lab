'''
1. Scopul acestui exercit, iu este de a calcula frecvent,ele prezente ˆın semnalul
prezentat ˆın Sect, iunea 2.
(a) Care este frecvent,a de es,antionare a semnalului din Train.csv (revedet, i
sect, iunea pentru detalii despre cum a fost achizit, ionat acesta)?
(b) Ce interval de timp acoper˘a es,antioanele din fis, ier?
(c) Considerˆand c˘a semnalul a fost es,antionat corect (f˘ar˘a aliere) s, i optim, care este frecvent,a maxim˘a prezent˘a ˆın semnal?
(d) Utilizat, i funct, ia np.fft.fft(x) pentru a calcula transformata Fourier a semnalului s, i afis,at, i grafic modulul transformatei.
Deoarece valorile pe care le vet, i calcula sunt ˆın Hz, este important
s˘a definit, i corect frecvent,a de es,antionare (astfel ˆıncˆat valorile de
frecvent,e pe care le obt, inet, i utilizˆand ultima secvent,˘a de cod din
Sect, iunea 3 s˘a aib˘a interpretare corect˘a din punct de vedere fizic).
(e) Prezint˘a acest semnal o component˘a continu˘a? Dac˘a da, eliminat, i-o.
Dac˘a nu, specificat, i cum at, i determinat.
(f) Care sunt frecvent,ele principale cont, inute ˆın semnal, as,a cum apar ele
ˆın transformata Fourier? Mai exact, determinat, i primele 4 cele mai
mari valori ale modulului transformatei s, i specificat, i c˘aror frecvent,e
(ˆın Hz) le corespund. C˘aror fenomene periodice din semnal se asociaz˘a fiecare?
(g) ˆIncepˆand de la es,antion ales de voi mai mare decˆat 1000, vizualizat, i,
pe un grafic separat, o lun˘a de trafic. Aleget, i es,antionul de start
astfel ˆıncˆat reprezentarea s˘a ˆınceap˘a ˆıntr-o zi de luni.
(h) Nu se cunoas, te data la care a ˆınceput m˘asurarea acestui semnal.
Concepet, i o metod˘a (descriet, i ˆın cuvinte) prin care s˘a determinat, i,
doar analizˆand semnalul ˆın timp, aceast˘a dat˘a. Comentat, i ce neajunsuri ar putea avea solut, ia propus˘a s, i care sunt factorii de care
depinde acuratet,ea ei.
(i) Filtrat, i semnalul, eliminat, i componentele de frecvent,˘a ˆınalt˘a (la alegerea voastr˘a care/cˆate, dar alegerea s˘a se poat˘a justifica).
'''

'''
Answers
a) "din ora-n ora" -> T = 1h = 3600s -> f = 1/3600 Hz

b) Fisierul contine 18288 sample-uri "din ora-n ora"
    -> 18288h = 762zile ~ 2 ani

c) 0.00013887369981530087

d) figures/ex1_d.pdf

e) Media semnalului != 0 -> contine o componenta continua
    figures/ex1_e.pdf

f) Biggest 4 frequencies: [0.00013883 0.00013884 0.00013886 0.00013887]

g) figures/ex1_g.pdf

h) 
    If we know the sample rate, like in this example, we can determine larger time units such as days and weeks.
    We can seek for patterns in:
        - the day, such as low traffic at night
        - the week, low traffic in the weekends
    Maybe we can observe a behaviour in the months, like low traffic in the summers or winters
    
    Problems:
        The patterns are pure assumptions with no evidence to support them and the found patterns might
        apply to different times, for example, the assumption of low traffic in the weekend might be 
        actually low traffic Mondey and Tuesday
    
i) figures/ex1_i.pdf
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_path = './datasets/Train.csv'
figures_directory = './figures'


if __name__ == "__main__":
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory, exist_ok=True)

    x = np.genfromtxt(dataset_path, delimiter=',')
    sample_rate = 1/3600
    samples = x[:, :1][1:].flatten()
    wave = x[:, 2:3][1:].flatten()

    date_times = pd.read_csv(dataset_path, parse_dates=["Datetime"], dayfirst=True)

    fft_components = np.fft.fft(wave)
    fft_frequencies = np.fft.fftfreq(wave.shape[0], 1/sample_rate)


    # c)
    # # 0.00013887369981530087
    print(f'c) Max frequency: {np.max(fft_frequencies)}')

    # d)
    # plot transformation
    fig, ax = plt.subplots(figsize=(12, 6), nrows=1, ncols=2)

    # plot stem
    markerline, stemlines, baseline = ax[0].stem(
        fft_frequencies, np.abs(fft_components), linefmt="k-", markerfmt="ko"
    )
    markerline.set_markerfacecolor("none")
    stemlines.set_linewidth(0.5)

    ax[0].set_yscale('log')
    ax[0].set_xscale('log')
    ax[0].set_xlim([0, np.max(fft_frequencies)])

    # plot plot
    ax[1].plot(fft_frequencies, np.abs(fft_components))
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].grid(True)
    fig.savefig(f"./{figures_directory}/ex1_d.pdf")
    plt.clf()

    # e)
    dc = np.mean(wave)

    plt.plot(samples, wave-dc)
    plt.grid(True)
    plt.title('Normalized wave')
    plt.savefig(f"./{figures_directory}/ex1_e.pdf")
    plt.clf()

    # f)
    nof_first_components = 4
    time_period = {
        "day": 3600 * 24,
        "week": 3600 * 24 * 7,
        "month": 3600 * 24 * 30,
        "year": 3600 * 24 * 365
    }

    normalized_fft = np.fft.fft(wave-dc)
    fft_frequencies = np.fft.fftfreq(wave.shape[0], 1 / sample_rate)[: len(normalized_fft)//2]
    sorted_magnitudes = np.argsort(fft_frequencies)

    top_magnitudes = fft_frequencies[sorted_magnitudes][-nof_first_components:]
    print(f'f) Biggest {nof_first_components} frequencies: {top_magnitudes}')

    for time in time_period:
        print(f'By {time}: {top_magnitudes * time_period[time]}')

    # g)
    nof_samples = 1111
    times = date_times.iloc[nof_samples:]
    monday_index = times[times["Datetime"].dt.weekday == 0].index[24]
    plt.plot(samples[monday_index:monday_index + nof_samples], wave[monday_index: monday_index + nof_samples])
    plt.grid(True)
    plt.title("One Month of Traffic from Monday")
    plt.savefig(f"./{figures_directory}/ex1_g.pdf")
    plt.clf()

    # h) answers are up
    # i)
    wave = wave[:750]
    samples = samples[:750]
    fft_components = np.fft.fft(wave)
    threshold = np.average(np.abs(fft_components)) * 1.5
    fft_filtered = [0+0j if np.abs(comp) < threshold else comp for comp in fft_components]

    filtered_signal = np.real(np.fft.ifft(fft_filtered))

    fig, ax = plt.subplots(nrows=2, ncols=1)
    y_max = max(np.max(np.abs(filtered_signal[:len(fft_filtered) // 2])), np.max(np.abs(wave[:len(fft_filtered) // 2])))

    ax[1].plot(samples[:len(samples)//2], filtered_signal[:len(fft_filtered)//2])
    ax[1].set_ylim(0, y_max)
    ax[1].grid(True)
    ax[1].set_title("Filtered signal")

    ax[0].plot(samples[:len(samples)//2], wave[:len(fft_filtered)//2])
    ax[0].set_ylim(0, y_max)
    ax[0].grid(True)
    ax[0].set_title("Initial signal")
    fig.tight_layout()
    fig.savefig(f"./{figures_directory}/ex1_i.pdf")

    plt.clf()