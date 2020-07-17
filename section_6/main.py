"""
@leofansq
https://github.com/leofansq
"""
import cv2
import numpy as np
import pywt

import matplotlib.pyplot as plt

def gasuss_noise(img, mean=0, var=0.001, alpha=0.1):
    """
    添加高斯噪声
    Parameters:
        mean : 噪声均值 
        var : 噪声方差
    """
    noise = np.random.normal(mean, var ** 0.5, img.shape)*255
    img_noise = img + alpha*noise
    img_noise = np.clip(img_noise, 0, 255.0)

    return img_noise

def denoise(img):
    """
    维纳滤波去噪
    Parameter:
        img: 含噪声图像
    Return:
        img_denoised: 维纳滤波去噪结果
    """
    # 小波分解
    A, (H, V, D) = pywt.dwt2(img, 'bior4.4')

    # 排版: 以便后续可视化小波分解结果
    AH = np.concatenate([A, H], axis=1)
    VD = np.concatenate([V, D], axis=1)
    fig = np.concatenate([AH, VD], axis=0)

    # 维纳滤波
    sigma_n = np.median(np.abs(D))/0.6745
   
    AHVD = []
    for i in [A, H, V, D]:
        sigma_sq = np.mean(i**2) - sigma_n**2
        i = (sigma_sq/(sigma_sq + sigma_n**2)) * i
        AHVD.append(i)
    [A, H, V, D] = AHVD
    
    # 小波重构
    img_denoised = pywt.idwt2((A,(H,V,D)), 'bior4.4')
    
    # 比较滤波前后差异
    img_diff = img -img_denoised

    # 可视化结果
    plt.figure("Wavelet Denoising")
    plt.subplot(2,2,1)
    plt.title("Image with Noise")
    plt.imshow(img, "gray")
    plt.subplot(2,2,2)
    plt.title("Wavelet Decomposition")
    plt.imshow(fig, "gray")
    plt.subplot(2,2,3)
    plt.title("Wavelet Denoised")
    plt.imshow(img_denoised, "gray")
    plt.subplot(2,2,4)
    plt.title("Difference")
    plt.imshow(img_diff, "gray")
    plt.tight_layout()
    plt.savefig("img_denoised.png")

    return img_denoised

def de(img, n, sigma_n=None):
    """
    维纳滤波去噪: 递归多层分解去噪
    Parameters:
        img: 待分解去噪图像
        n: 待分解层数
        sigma_n: 噪声方差(由第一层分解的HH计算得到)
    Return:
        img_denoised: 维纳滤波去噪结果
    """
    # 递归终止条件: 待分解层数为0
    if not(n): return img

    # 递归多层维纳滤波
    # 小波分解
    A, (H, V, D) = pywt.dwt2(img, 'bior4.4')
    # 对于第一层分解, 计算噪声方差
    if not(sigma_n): sigma_n = np.median(np.abs(D))/0.6745
    # 递归
    A = de(A, n-1, sigma_n)
    # 递归异常处理: 处理pywt对奇数行列图像分解重构后行列数增加1的特殊情况
    if A.shape[0] > H.shape[0]: A = A[:-1, :-1]
    # 维纳滤波
    AHVD = []
    for i in [H, V, D]:
        sigma_sq = np.mean(i**2) - sigma_n**2
        i = (sigma_sq/(sigma_sq + sigma_n**2)) * i
        AHVD.append(i)
    [H, V, D] = AHVD
    # 小波重构
    img_denoised = pywt.idwt2((A,(H,V,D)), 'bior4.4')

    return img_denoised

if __name__ == "__main__":
    # 读入图像
    img = cv2.imread("lena.png", 0)
    # 添加高斯噪声
    img_noise = gasuss_noise(img, var=0.4, alpha=0.25)

    plt.figure("Add Gaussian Noise")
    plt.subplot(1,2,1)
    plt.title("Origin Image")
    plt.imshow(img, "gray")
    plt.subplot(1,2,2)
    plt.title("Image with Noise")
    plt.imshow(img_noise, "gray")
    plt.savefig("img_noise.png")

    # 单层维纳滤波去噪实验
    img_de = denoise(img_noise)

    # 多层维纳滤波去噪效果对比试验
    N = 5
    plt.figure("Multi-decomposition")
    plt.subplot(2,3,1)
    plt.title("Image with Noise")
    plt.imshow(img_noise, "gray")
    for i in range(N):
        img_denoised = de(img_noise, i+1)

        plt.subplot(2,3,i+2)
        plt.title("{} Decomposition".format(i+1))
        plt.imshow(img_denoised, "gray")
    plt.tight_layout()
    plt.savefig("multi-decomposition.png")

