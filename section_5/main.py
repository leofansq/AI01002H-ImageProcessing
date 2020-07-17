"""
@leofansq
https://github.com/leofansq
"""
import cv2
import numpy as np

from binaryzation import binaryzation 
from sk_morph import sk_morph
from sk_thin import sk_thin
from sk_distTrans import sk_distTrans
from tailor import tailor

def main(img):
    """
    主程序
    """
    # 二值化: 基于迭代法获取二值化阈值
    print ("Processing: 二值化...")
    k, img_bin = binaryzation(img)
    print ("阈值:", k)
    cv2.imwrite("bin.png", img_bin)

    # 基于腐蚀和开运算的骨架提取
    print ("Processing: 基于腐蚀和开运算的骨架提取...")
    img_sk_morph = sk_morph(img_bin)
    cv2.imwrite("morph.png", img_sk_morph)

    # 基于单纯细化的骨架提取
    print ("Processing: 基于单纯细化的骨架提取...")
    img_sk_thin = sk_thin(img_bin)
    cv2.imwrite("morph_thin.png", img_sk_thin)

    # 基于距离变换的骨架提取
    print ("Processing: 基于距离变换的骨架提取...")
    img_sk_dist = sk_distTrans(img_bin)
    cv2.imwrite("dist.png", img_sk_dist)

    # 裁剪:以细化所得骨架为例
    print ("Processing: 裁剪:以细化所得骨架为例...")
    img_result = tailor(img_sk_thin)
    cv2.imwrite("tailor.png", img_result)

if __name__ == "__main__":
    img = cv2.imread("origin.png", 0)
    main(img)