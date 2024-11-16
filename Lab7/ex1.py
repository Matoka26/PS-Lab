import matplotlib.pyplot as plt
import numpy as np
import os


figures_dir = './figures'

if __name__ == "__main__":
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir, exist_ok=True)

    # a)
    nof_pixels = 128
    u = 2 * np.pi
    v = 3 * np.pi
    x_samples = np.arange(0, nof_pixels)
    y_samples = np.arange(0, nof_pixels)

    img = np.zeros((nof_pixels, nof_pixels))

    for i, x in enumerate(x_samples):
        for j, y in enumerate(y_samples):
            img[i][j] = np.sin(u * x + v * y)

    plt.imshow(img, cmap='grey')
    plt.savefig(f"./{figures_dir}/ex1_a.pdf")
    plt.clf()

    # b)
    u = 4 * np.pi
    v = 6 * np.pi
    x_samples = np.arange(0, nof_pixels)
    y_samples = np.arange(0, nof_pixels)

    for i, x in enumerate(x_samples):
        for j, y in enumerate(y_samples):
            img[i][j] = np.sin(u * x) + np.cos(v * y)

    plt.imshow(img, cmap='grey')
    plt.savefig(f"./{figures_dir}/ex1_b.pdf")
    plt.clf()

    # c)
    slice_ind = 5
    fft_2d = np.zeros((nof_pixels, nof_pixels))
    fft_2d[0][slice_ind], fft_2d[0][nof_pixels-slice_ind] = 1, 1
    img = np.real(np.fft.ifft2(fft_2d))

    plt.imshow(img, cmap='grey')
    plt.savefig(f"./{figures_dir}/ex1_c.pdf")
    plt.clf()

    # d)
    slice_ind = 5
    fft_2d = np.zeros((nof_pixels, nof_pixels))
    fft_2d[slice_ind][0], fft_2d[nof_pixels - slice_ind][0] = 1, 1
    img = np.real(np.fft.ifft2(fft_2d))

    plt.imshow(img, cmap='grey')
    plt.savefig(f"./{figures_dir}/ex1_d.pdf")
    plt.clf()

    # e)
    slice_ind = 5
    fft_2d = np.zeros((nof_pixels, nof_pixels))
    fft_2d[slice_ind][0], fft_2d[nof_pixels - slice_ind][0] = 1, 1
    fft_2d[0][slice_ind], fft_2d[0][nof_pixels-slice_ind] = 1, 1
    img = np.real(np.fft.ifft2(fft_2d))

    plt.imshow(img, cmap='grey')
    plt.savefig(f"./{figures_dir}/ex1_e.pdf")
    plt.clf()