"""
@leofansq
https://github.com/leofansq
"""
import cv2
import numpy as np

from basic_function import dilate

def thinning(img, K):
    """
    细化运算实体
    Parameters:
        img: 待细化图像
        K: 结构子序列
    Return:
        细化后结果图像
    """
    # 归一
    img_result = img/255
    # 初始化用于保存上一次结果的矩阵
    img_old = 1 - img_result

    # 循环细化.直至图像保持不变
    while np.sum(img_result-img_old):
        img_old = img_result
        for i in K:
            # 基于卷积结果的击中击不中
            img_temp = np.where(cv2.filter2D(img_result.copy(),-1,i,borderType=0)==15, 1, 0)            
            img_result = img_result - img_temp
   
    img_result *= 255
    return img_result.astype(np.uint8)

def sk_thin(img):
    """
    细化提取骨架
    Parameter:
        img: 待提取图像
    Return:
        提取骨架结果图像 
    """
    # 生成8个结构子序列
    k_1 = np.array([[16,16,16],[0,1,0],[2,4,8]], dtype=np.uint8)
    k_2 = np.array([[0,16,16],[1,2,16],[4,8,0]], dtype=np.uint8)
    k_3 = np.array([[1,0,16],[2,4,16],[8,0,16]], dtype=np.uint8)
    k_4 = np.array([[1,2,0],[4,8,16],[0,16,16]], dtype=np.uint8)
    k_5 = np.array([[1,2,4],[0,8,0],[16,16,16]], dtype=np.uint8)
    k_6 = np.array([[0,1,2],[16,4,8],[16,16,0]], dtype=np.uint8)
    k_7 = np.array([[16,0,1],[16,2,4],[16,0,8]], dtype=np.uint8)
    k_8 = np.array([[16,16,0],[16,1,2],[0,4,8]], dtype=np.uint8)

    K = [k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8]
    
    # 细化操作
    img_result = thinning(img, K)

    return img_result


if __name__ == "__main__":
    # 测试用图
    # img = 255 - np.zeros((5, 11))
    # img[1:5,9] = img[1:5,10] = img[4,3:5] = 0

    img = cv2.imread("bin.png", 0)
    img_result = sk_thin(img)
    cv2.imwrite("morph_thin.png", img_result)