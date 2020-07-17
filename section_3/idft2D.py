"""
@leofansq
https://github.com/leofansq
"""
import cv2
import numpy as np

def idft2D(F):
    """
    实现二维傅里叶逆变换
    Parameters:
        F : 灰度图像的傅里叶变换结果
    Return:
        傅里叶逆变换结果
    """
    # Conjugate
    f = F.conjugate()
    
    # 2D FFT
    for i in range(f.shape[0]):
        f[i] = np.fft.fft(f[i])
    for i in range(f.shape[1]):
        f[:, i] = np.fft.fft(f[:, i])
    
    # Divide by MN & Conjugate
    f = f/f.size
    f = np.abs(f.conjugate())

    return f


if __name__ == "__main__":
    # img = cv2.imread("house.tif", cv2.IMREAD_GRAYSCALE)
    img = np.zeros([512,512])
    img[226:285, 251:260] = 255

    img = img/255.
    
    from dft2D import dft2D
    img_fft = dft2D(img)

    img_ifft = idft2D(img_fft)

    import matplotlib.pyplot as plt
    plt.subplot(141)
    plt.imshow(img, cmap='gray')
    plt.subplot(142)
    plt.imshow(np.round(np.abs(img_fft)), cmap='gray')
    plt.subplot(143)
    plt.imshow(np.abs(img_ifft), cmap='gray')
    plt.subplot(144)
    plt.imshow(np.round(img - img_ifft), cmap='gray')
    plt.show()