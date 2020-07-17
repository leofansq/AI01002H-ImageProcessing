"""
@leofansq
https://github.com/leofansq
"""
import cv2
import numpy as np

def Lap_eh(img):
    """
    拉普拉斯增强
    Parameters:
        img: 待增强图像
    Return:
        增强后的图像
    """
    lap_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_eh = cv2.filter2D(img, -1, lap_kernel)

    return img_eh

if __name__ == "__main__":
    img = cv2.imread("luna.png", 0)
    img_eh = Lap_eh(img)
    cv2.imwrite("pb3_result.png", img_eh)