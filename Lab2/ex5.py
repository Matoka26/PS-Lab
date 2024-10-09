'''
5. Generat, i dou˘a semnale cu aceeas, i form˘a de und˘a, dar de frecvent,e diferite,
s, i punet, i-le unul dup˘a cel˘alalt ˆın acelas, i vector. Redat, i audio rezultatul s, i
notat, i ce observat, i.

'''


import os
import numpy as np
import sounddevice as sd
from scipy.io import wavfile

wave_sounds_directory = 'wave_sounds'


def play_wave(wave: np.array, file_name: str, audio_sample_rate: int = 44100, loop: bool = False) -> None:
    if not os.path.isdir(wave_sounds_directory):
        os.makedirs(wave_sounds_directory, exist_ok=True)

    wavfile.write(f'./{wave_sounds_directory}/{file_name}.wav', audio_sample_rate, wave)
    sd.play(wave, samplerate=audio_sample_rate, loop=loop)     # Make loop=True to actually hear it, it ends too quick
    sd.wait()


def sinusoidal(t: float, f=1, a=1, phi=0) -> float:
    return a * np.sin(2 * np.pi * f * t + phi)


if __name__ == "__main__":
    if not os.path.isdir(wave_sounds_directory):
        os.makedirs(wave_sounds_directory, exist_ok=True)

    nof_samples = 1000
    f1 = 10
    f2 = 100
    a = 1
    phi = np.pi

    samples = np.linspace(0, 1, nof_samples)
    wave1 = sinusoidal(samples, f=f1, a=a, phi=phi)
    wave2 = sinusoidal(samples, f=f2, a=a, phi=phi)
    concatenated_wave = np.concatenate((wave1, wave2))
    play_wave(concatenated_wave, file_name='ex5_concatenated_waves', loop=True)
