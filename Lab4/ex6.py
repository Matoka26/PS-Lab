from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import os

# Great thanks to skpha13 for helping me on this one

song_path = 'songs_mp3/song.wav'
figures_directory1 = './figures_pdf'
figures_directory2 = './figures_png'


def get_segmented_wave(data: np.ndarray) -> np.ndarray:
    segmented_data = []
    overlap_size = len(data) // 100  # for %1 length from initial signal
    step = overlap_size // 2         # for 50% overlap

    for i in range(0, len(data) - overlap_size + 1, step):
        segmented_data.append(data[i: i + overlap_size])

    return np.array(segmented_data)


if __name__ == '__main__':
    if not os.path.isdir(figures_directory1):
        os.makedirs(figures_directory1, exist_ok=True)
    if not os.path.isdir(figures_directory2):
        os.makedirs(figures_directory2, exist_ok=True)

    # read song
    sample_rate, audio_data = wavfile.read(song_path)
    # crop a random portion of the song
    audio_data = audio_data[4000:5000]

    segmented_audio = get_segmented_wave(audio_data)
    fft_signal = np.fft.fft(segmented_audio, axis=1)

    spectrogram = np.transpose(
        # select all rows, and half the columns do to symmetry
        np.abs(fft_signal)[:, : fft_signal.shape[1] // 2]
    )

    # add small constant to avoid log(0)
    spectrogram = np.log10(spectrogram + 1e-10)
    print(spectrogram)

    # extract the samples frequencies
    frequencies = np.fft.fftfreq(segmented_audio.shape[1], d=1 / sample_rate)[: fft_signal.shape[1] // 2]
    print(frequencies)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        spectrogram,
        aspect="auto",
        extent=(0, len(audio_data) / sample_rate, frequencies[0], frequencies[-1]),
        origin="lower",
        cmap="magma",
        vmin=0,
        vmax=100,
    )
    plt.colorbar(label="Magnitude")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (kHz)")

    name = "Spectrogram of Audio Signal"
    plt.title(name)
    plt.savefig(f"plots/{name}.pdf", format="pdf")
    plt.savefig(f"plots/{name}.png", format="png")

    plt.show()