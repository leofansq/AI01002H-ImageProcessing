"""
@leofansq
https://github.com/leofansq
"""
import cv2
import numpy as np

def loop(img):
    """
    根据阈值分割的迭代法计算阈值T
    Parameters:
        img: origin image (GRAY)
    Return:
        Threshold T
    """
    g_min = int(np.min(img))
    g_max = int(np.max(img))
    T = 0
    T_new = 0.5*(g_min + g_max)

    eps = 1e-5

    while np.abs(T_new - T) >= eps:
        T = T_new
        l = []
        g = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i,j] < T: l.append(img[i,j])
                else: g.append(img[i,j])
        T_new = 0.5*(np.mean(l) + np.mean(g))
    
    return T_new

def binaryzation(img):
    """
    灰度图像二值化
    Parameter:
        img: 灰度图像
    Return:
        二值化图像
    """
    # 迭代法求阈值
    k = loop(img)
    # 二值化
    img_bin = np.where(img>k, 0, 255)
    
    return k, img_bin.astype(np.uint8)

if __name__ == "__main__":
    img = cv2.imread("origin.png", 0)
    k = loop(img)
    print (k)
    _,img_bin = cv2.threshold(img.copy(), k, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("bin.png", img_bin)

    
