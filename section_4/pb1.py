"""
@leofansq
https://github.com/leofansq
"""
import cv2
import numpy as np 

def histequal4e(img):
    """
    直方图均衡化
    Parameters:
        img : 原图片(GRAY)
    Return:
        直方图均衡化后的图片
    """
    # 获取原图像灰度直方图
    hist = np.bincount(img.flatten(), minlength=256)

    # 根据比重构建均衡化后的直方图
    hist_new = np.cumsum(hist)/np.sum(hist) * 255

    # 生成直方图均衡化的图片
    img_result = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_result[i,j] = hist_new[img[i,j]]
    
    return img_result

if __name__ == "__main__":
    img = cv2.imread("luna.png", 0)
    img_result = histequal4e(img)
    cv2.imwrite("pb1_result.png", img_result)


