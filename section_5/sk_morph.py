"""
@leofansq
https://github.com/leofansq
"""
import cv2
import numpy as np

from basic_function import erode, open_morph
    
def sk_morph(img):
    """
    形态学骨架提取
    Parameter:
        img: 待提取骨架图像(默认为前景为白色的二值图像)
    Return:
        img_result: 骨架图像(前景为白色的二值图像)
    """
    # 骨架图像初始化
    img_result = np.zeros_like(img)

    # 循环提取骨架, 当腐蚀后图像无前景时停止
    while(np.sum(img)):
        # 开运算
        img_open = open_morph(img)
        # 求差
        img_s = img - img_open
        # 求并生成骨架
        img_result = cv2.bitwise_or(img_result, img_s.copy())      
        # 腐蚀
        img = erode(img)
    return img_result
    

if __name__ == "__main__":
    img = cv2.imread("bin.png", 0)
    img_result = sk_morph(img)
    cv2.imwrite("morph.png", img_result)

