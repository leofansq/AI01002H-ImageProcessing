"""
@leofansq
https://github.com/leofansq
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

from dft2D import dft2D
from idft2D import idft2D

def main(filepath):
    """
    Main function for problem 3.
    """
    # Read the image
    img = cv2.imread(filepath, 0)
    img = img/255.

    # 2D FFT
    img_fft = dft2D(img)

    # 2D IFFT 
    img_ifft = idft2D(img_fft)

    # Cal difference
    img_diff = img - img_ifft

    # Plot the result
    plt.subplot(221).set_title("Origin")
    plt.imshow(img, cmap='gray')

    plt.subplot(222).set_title("FFT")
    plt.imshow(np.log(np.abs(img_fft)+1), cmap='gray')

    plt.subplot(223).set_title("IFFT")
    plt.imshow(img_ifft, cmap='gray')

    plt.subplot(224).set_title("Difference")
    plt.imshow(np.round(img_diff), cmap='gray')

    plt.tight_layout()
    plt.savefig("problem3.png")

if __name__ == "__main__":
    main("rose512.tif")
