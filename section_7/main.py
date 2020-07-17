"""
@leofansq
https://github.com/leofansq
"""
import cv2
import numpy as np 
import math

import matplotlib.pyplot as plt

def blurry_analysis(img_origin, img_blur):
    """
    可参数化的模糊核估计 & 可视化对比
    Parameter:
        img_origin: Origin image
        img_blur: Blurry image
    """
    # 模糊核估计
    img_b_fft = np.fft.fft2(np.log(np.fft.fft2(img_blur)))
    img_b_fft = np.fft.fftshift(img_b_fft)
    img_b_fft = 20*np.log(np.abs(img_b_fft))
    # 对原图做相同操作, 用以对比
    img_o_fft = np.fft.fft2(np.log(np.fft.fft2(img_origin)))
    img_o_fft = np.fft.fftshift(img_o_fft)
    img_o_fft = 20*np.log(np.abs(img_o_fft))
    # 取中间部分图像, 便于可视化观察
    h, w = img_o_fft.shape[:2]
    img_o_center = img_o_fft[int(3*h/10):int(7*h/10), int(3*w/10):int(7*w/10)]
    img_b_center = img_b_fft[int(3*h/10):int(7*h/10), int(3*w/10):int(7*w/10)]
    # 可视化
    plt.subplot(2,3,1)
    plt.title("Origin Image")
    plt.imshow(img_origin, cmap='gray')
    plt.subplot(2,3,2)
    plt.title("Analysis Result")
    plt.imshow(img_o_fft, cmap='gray')
    plt.subplot(2,3,3)
    plt.title("Center of Result")
    plt.imshow(img_o_center, cmap='gray')
    plt.subplot(2,3,4)
    plt.title("Blurry Image")
    plt.imshow(img_blur, cmap='gray')
    plt.subplot(2,3,5)
    plt.title("Analysis Result")
    plt.imshow(img_b_fft, cmap='gray')
    plt.subplot(2,3,6)
    plt.title("Center of Result")
    plt.imshow(img_b_center, cmap='gray')
    plt.tight_layout()
    plt.savefig("analysis.png")

def kernel_est(img_shape, theta, length):
    """
    运动模糊核重建
    Parameter:
        img_shape: 图像尺寸[h,w]
        theta: 运动模糊核角度
        length: 运动模糊核长度
    Return:
        kernel: 重建的运动模糊核
    """
    # 参数初始化
    h, w = img_shape[:2]
    pos_c_h = round(h/2)
    pos_c_w = round(w/2)
    theta = np.pi * (-theta/180)
    # 重建运动模糊核
    kernel = np.zeros((h, w))
    for i in range(length):
        l = length/2 - i
        delta_w = l * np.cos(theta)
        delta_h = l * np.sin(theta)
        kernel[int(pos_c_w+delta_w), int(pos_c_h+delta_h)] = 1
    kernel = kernel/np.sum(kernel)

    return kernel

def get_blurry_k_gt(img_origin, img_blur):
    """
    获取真实的模糊核
    Parameter:
        img_origin: 原始清晰图像
        img_blur: 模糊图像
    Return:
        blurry_fft: 真实的模糊核
    """
    img_o_fft = np.fft.fft2(img_origin)
    img_b_fft = np.fft.fft2(img_blur)
    blurry_fft = np.fft.ifft2(img_b_fft / img_o_fft)
    blurry_fft = np.abs(np.fft.fftshift(blurry_fft))
    blurry_fft = blurry_fft/ np.sum(blurry_fft)

    return blurry_fft
        
def wiener(img_blur, kernel, K=0.005):
    """
    维纳滤波
    Parameter:
        img_blur: 模糊图像
        kernel: 模糊核
        K: 分母参数
    Return:
        img_result: 维纳滤波去模糊结果
    """
    img_b_fft = np.fft.fft2(img_blur)
    kernel_fft = np.fft.fft2(kernel)
    kernel_fft = np.conj(kernel_fft) / (np.abs(kernel_fft)**2 + K)
    img_result = np.fft.ifft2(img_b_fft * kernel_fft)
    img_result = np.abs(np.fft.fftshift(img_result))
    img_result = img_result.astype(np.uint8)

    return img_result

if __name__ == "__main__":
    # 读入图像
    img = cv2.imread("./lena.png", 0)
    img_blur = cv2.imread("./test.png", 0)

    # 可参数化模糊核估计 & 可视化
    blurry_analysis(img, img_blur)
    
    # 模糊核重建 & 对比
    blurry_k = kernel_est(img_blur.shape, -30, 42)
    blurry_k_gt = get_blurry_k_gt(img,  img_blur)

    plt.subplot(1,2,1)
    plt.title("blurry Kernel GroundTruth")
    plt.imshow(blurry_k_gt, cmap='gray')
    plt.subplot(1,2,2)
    plt.title("blurry Kernel Est-result")
    plt.imshow(blurry_k, cmap='gray')
    plt.tight_layout()
    plt.savefig("blurry_kernel.png")

    # 维纳滤波去模糊
    img_result = wiener(img_blur, blurry_k)

    plt.subplot(1,3,1)
    plt.title("Origin Image")
    plt.imshow(img, cmap='gray')
    plt.subplot(1,3,2)
    plt.title("Blurry Image")
    plt.imshow(img_blur, cmap='gray')
    plt.subplot(1,3,3)
    plt.title("Deblur Result")
    plt.imshow(img_result, cmap='gray')
    plt.tight_layout()
    plt.savefig("result.png")

    # Edgetaper 去振铃效果对比
    img_et = cv2.imread("./lena_edgetaper.png", 0)
    img_et_result = wiener(img_et, blurry_k)

    plt.subplot(2,2,1)
    plt.title("Origin Image")
    plt.imshow(img, cmap='gray')
    plt.subplot(2,2,2)
    plt.title("Blurry Image")
    plt.imshow(img_blur, cmap='gray')
    plt.subplot(2,2,3)
    plt.title("Deblur Result")
    plt.imshow(img_result, cmap='gray')
    plt.subplot(2,2,4)
    plt.title("Edgetaper-Deblur")
    plt.imshow(img_et_result, cmap='gray')
    plt.tight_layout()
    plt.savefig("result_et.png")
