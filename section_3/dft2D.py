"""
@leofansq
https://github.com/leofansq
"""
import cv2
import numpy as np

def dft2D(f):
    """
    通过计算一维傅里叶变换实现图像二维快速傅里叶变换
    Parameters:
        f: image (Gray scale)
    Return:
        the result of FFT
    """
    
    F = f.copy()
    F = F.astype(np.complex128)

    # FFT for each row
    for i in range(F.shape[0]):
        F[i] = np.fft.fft(F[i])
    
    # FFT for each column
    for i in range(F.shape[1]):
        F[:, i] = np.fft.fft(F[:, i])
    
    return F

if __name__ == "__main__":
    img = np.zeros([512,512])
    img[226:285, 251:260] = 255

    img = img/255.

    img_fft = dft2D(img)

    import matplotlib.pyplot as plt
    plt.imshow(np.abs(img_fft), cmap='gray')
    plt.show()
    
