"""
@leofansq
https://github.com/leofansq
"""
import cv2
import numpy as np

from basic_function import dilate

def thinning(img, K):
    """
    细化
    Parameters:
        img: 待细化图像
        K: 结构子序列
    Return:
        细化后结果图像
    """
    # 归一
    img_result = img/255
    # 利用结构子序列重复3次细化
    for i in range(3):
        for i in K:
            img_temp = np.where(cv2.filter2D(img_result.copy(),-1,i,borderType=0)==3, 1, 0)
            img_result = img_result - img_temp
    
    img_result *= 255
    return img_result.astype(np.uint8)

def find_end(img, K):
    """
    找到端节点
    Parameters:
        img: 输入图像
        K: 结构子序列
    Return:
        只有端节点为前景的图像
    """
    # 像素归一化
    img_ones = img/255
    img_result = np.zeros_like(img, dtype=np.uint8)

    # 利用结构子序列寻找端点
    for i in K:
        img_temp = np.where(cv2.filter2D(img_ones.copy(),-1,i,borderType=0)==3, 1, 0)
        img_result = img_result + img_temp
    
    img_result *= 255
    return img_result.astype(np.uint8)

def tailor(img):
    """
    裁剪
    Parameters:
        img: 待裁剪图像
    Return:
        裁剪结果图像
    """
    # 生成8个结构子
    k_1 = np.array([[0,4,4],[1,2,4],[0,4,4]], dtype=np.uint8)
    k_2 = np.array([[0,1,0],[4,2,4],[4,4,4]], dtype=np.uint8)
    k_3 = np.array([[4,4,0],[4,1,2],[4,4,0]], dtype=np.uint8)
    k_4 = np.array([[4,4,4],[4,1,4],[0,2,0]], dtype=np.uint8)
    k_5 = np.array([[1,4,4],[4,2,4],[4,4,4]], dtype=np.uint8)
    k_6 = np.array([[4,4,1],[4,2,4],[4,4,4]], dtype=np.uint8)
    k_7 = np.array([[4,4,4],[4,1,4],[4,4,2]], dtype=np.uint8)
    k_8 = np.array([[4,4,4],[4,1,4],[2,4,4]], dtype=np.uint8) 

    K = [k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8]

    # 细化(去除3个像素组成的分支)
    img_thin = thinning(img, K)
    # 找端点
    img_end = find_end(img_thin, K)
    # 膨胀运算,捡回误伤元素
    img_dilate = img_end
    for _ in range(3):
        img_dilate = dilate(img_dilate)
        img_dilate = cv2.bitwise_and(img_dilate, img)
    # 获得裁剪结果
    img_result = cv2.bitwise_or(img_dilate, img_thin)

    return img_result


if __name__ == "__main__":
    img = cv2.imread("morph_thin.png", 0)
    img_result = tailor(img)
    cv2.imwrite("tailor.png", img_result)