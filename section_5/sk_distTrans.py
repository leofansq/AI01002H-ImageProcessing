"""
@leofansq
https://github.com/leofansq
"""
import cv2
import numpy as np

from basic_function import erode

def find_max(img):
    """
    获得8邻域内极大值像素组成的图像
    Parameter:
        img: 待操作图像(距离变换结果)
    Return:
        img_result: 由8邻域内极大值像素组成的二值化图像
    """
    # 生成8个减法模板
    kmax_1 = np.array([[-1,0,0],[0,1,0],[0,0,0]],dtype=np.float32)
    kmax_2 = np.array([[0,-1,0],[0,1,0],[0,0,0]],dtype=np.float32)
    kmax_3 = np.array([[0,0,-1],[0,1,0],[0,0,0]],dtype=np.float32)
    kmax_4 = np.array([[0,0,0],[-1,1,0],[0,0,0]],dtype=np.float32)
    kmax_5 = np.array([[0,0,0],[0,1,-1],[0,0,0]],dtype=np.float32)
    kmax_6 = np.array([[0,0,0],[0,1,0],[-1,0,0]],dtype=np.float32)
    kmax_7 = np.array([[0,0,0],[0,1,0],[0,-1,0]],dtype=np.float32)
    kmax_8 = np.array([[0,0,0],[0,1,0],[0,0,-1]],dtype=np.float32)
    kernel = [kmax_1, kmax_2, kmax_3, kmax_4, kmax_5, kmax_6, kmax_7, kmax_8]
    
    # 依次进行减法模板操作, 取结果交集为极大值像素图像
    img_result = cv2.bitwise_not(np.zeros_like(img, dtype=np.uint8))
    for i in kernel:
        # 减法模板滤波
        img_m = cv2.filter2D(img, -1, i)
        # 差值非负处取为255: 操作点像素值>=被减处像素
        img_m = np.where(img_m>=0.0, 255, 0)
        img_m = img_m.astype(np.uint8)
        # 大于等于8邻域内所有像素的点为区域极大值点
        img_result = cv2.bitwise_and(img_result, img_m)
    
    return img_result

def sk_distTrans(img):
    """
    基于距离变换的骨架提取
    Parameter:
        img: 待提取骨架图像(默认为前景为白色的二值图像)
    Return:
        img_result: 骨架图像(前景为白色的二值图像)
    """
    # 通过形态学操作获得前景边界
    img_bd = img - erode(img)
    # cv2.imwrite("bd.png", img_bd)
    # 对边界图像做距离变换
    img_distTrans = cv2.distanceTransform(cv2.bitwise_not(img_bd.copy()), cv2.DIST_L2, cv2.DIST_MASK_3)
    # 求距离变换图中的局部极大值
    img_max = find_max(img_distTrans)
    # 落入原二值图像中的局部极大值即为图像的骨架
    img_result = cv2.bitwise_and(img_max, img)

    return img_result


if __name__ == "__main__":
    img = cv2.imread("bin.png", 0)
    img_result = sk_distTrans(img)
    cv2.imwrite("dist.png", img_result)

    