"""
@leofansq
https://github.com/leofansq
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

from dft2D import dft2D
from idft2D import idft2D

def main():
    """
    Main Function for problem 4.
    """
    # Generate the image
    img = np.zeros([512,512])
    img[226:285, 251:260] = 255
    img = img/255.

    # FFT
    img_fft = dft2D(img)

    # Centralization FFT
    img_fft_c = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_fft_c[i, j] = img_fft_c[i, j] * ((-1)**(i+j))

    img_fft_c = dft2D(img_fft_c)

    # Logarithmic transformation
    img_fft_clog = np.log(1 + np.abs(img_fft_c)) 

    # Plot
    plt.subplot(221).set_title("Origin")
    plt.imshow(img, cmap='gray')

    plt.subplot(222).set_title("FFT")
    plt.imshow(np.abs(img_fft), cmap='gray')

    plt.subplot(223).set_title("Centralization")
    plt.imshow(np.abs(img_fft_c), cmap='gray')

    plt.subplot(224).set_title("Log")
    plt.imshow(img_fft_clog, cmap='gray')

    plt.tight_layout()
    plt.savefig("problem4.png")

if __name__ == "__main__":
    main()