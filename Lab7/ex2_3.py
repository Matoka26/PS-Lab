import matplotlib.pyplot as plt
import numpy as np
from scipy import misc, ndimage
import os
import copy

figures_dir = './figures'


def filter_inner_circle(img: np.ndarray, radius: int = 50, inout: str = 'in') -> np.ndarray:
    dic = {
        "in": lambda x, y: (x - x_center) ** 2 + (y - y_center) ** 2 > radius ** 2,
        "out": lambda x, y: (x - x_center) ** 2 + (y - y_center) ** 2 < radius ** 2
    }
    if inout not in dic:
        raise ValueError('Invalid inout value.')

    mat = copy.deepcopy(img)
    rows, cols = mat.shape
    x_center, y_center = rows//2, cols//2

    for x in range(rows):
        for y in range(cols):
            # Check if the point (x, y) lies on the circle
            if dic[inout](x, y):
                mat[x][y] = 0

    return mat


if __name__ == "__main__":
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir, exist_ok=True)

    raccoon = misc.face(gray=True)

    # plot initial image
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].imshow(raccoon, cmap=plt.cm.gray)
    ax[0].set_title("Initial image")

    # plot spectrum
    fft_2 = np.fft.fft2(raccoon)
    fft_2 = np.fft.fftshift(fft_2)
    # fft_2[260:290] = 0
    fft_2 = filter_inner_circle(fft_2, radius=30, inout="in")      # 'cut' inner circle

    im1 = ax[1].imshow(
        10*np.log10(np.abs(fft_2) + 1e-6),     # Change to log sclae
        aspect="auto",
        cmap="inferno",
    )
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title("Filtered Spectrum")
    fig.colorbar(im1, ax=ax[1], fraction=0.04)

    # plot reconstructed image
    filtered_raccoon = np.real(np.fft.ifft2(np.fft.ifftshift(fft_2)))     # Reconstruct image
    ax[2].imshow(filtered_raccoon, cmap=plt.cm.gray)
    ax[2].set_title("Reconstructed image")

    fig.tight_layout()
    fig.savefig(f"./{figures_dir}/ex2_3.pdf")

    print(f"Initial SNR: {10 * np.log10(np.mean(raccoon) / np.std(raccoon) + 1e-6)}dB")
    print(f"Filtered SNR: {10 * np.log10(np.mean(filtered_raccoon) / np.std(filtered_raccoon) + 1e-6)}dB")
