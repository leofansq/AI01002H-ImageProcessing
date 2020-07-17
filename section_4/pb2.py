"""
@leofansq
https://github.com/leofansq
"""
import cv2
import numpy as np

def sp_noise(img):
    """
    给图像添加椒盐噪声
    Parameters:
        img: 原图
    Return:
        添加椒盐噪声后的图片
    """
    import random

    img_noise = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()
            # 椒噪声
            if rdn < 0.05:
                img_noise[i][j] = 0
            # 盐噪声
            elif rdn > 0.95:
                img_noise[i][j] = 255
            # 不添加噪声
            else:
                img_noise[i][j] = img[i][j]

    return img_noise

def smooth(img):
    """
    用选择保边缘平滑法
    Parameters:
        img: 待平滑图像(GRAY)
    Return:
        平滑后图像
    """
    img_smooth = np.zeros_like(img)
    h, w = img.shape[0:2]
    for i in range(h):
        for j in range(w):
            std_mean = []

            # 3邻域
            if i>0 and j>0 and i<h-1 and j<w-1:
                mask = img[i-1:i+2, j-1:j+2].flatten().tolist()
                std_mean.append((np.std(mask),np.mean(mask)))
            
            # 上五边形
            if i>1 and j>0 and j<w-1:
                mask = img[i-2:i, j-1:j+2].flatten().tolist()
                mask.append(img[i,j])
                std_mean.append((np.std(mask),np.mean(mask)))
            
            # 下五边形
            if j>0 and i<h-2 and j<w-1:
                mask = img[i+1:i+3, j-1:j+2].flatten().tolist()
                mask.append(img[i,j])
                std_mean.append((np.std(mask),np.mean(mask)))
            
            # 左五边形
            if i>0 and j>1 and i<h-1:
                mask = img[i-1:i+2, j-2:j].flatten().tolist()
                mask.append(img[i,j])
                std_mean.append((np.std(mask),np.mean(mask)))
            
            # 右五边形
            if i>0 and i<h-1 and j<w-2:
                mask = img[i-1:i+2, j+1:j+3].flatten().tolist()
                mask.append(img[i,j])
                std_mean.append((np.std(mask),np.mean(mask)))
            
            # 左上六边形
            if i>1 and j>1:
                mask = [img[i-2,j-2], img[i-2,j-1], img[i-1,j-2], img[i-1,j-1], img[i-1,j], img[i,j-1], img[i,j]]
                std_mean.append((np.std(mask),np.mean(mask)))
            
            # 右上六边形
            if i>1 and j<w-2:
                mask = [img[i-2,j+2], img[i-2,j+1], img[i-1,j+2], img[i-1,j+1], img[i-1,j], img[i,j+1], img[i,j]]
                std_mean.append((np.std(mask),np.mean(mask)))
            
            # 右下六边形
            if i<h-2 and j<w-2:
                mask = [img[i+2,j+2], img[i+2,j+1], img[i+1,j+2], img[i+1,j+1], img[i+1,j], img[i,j+1], img[i,j]]
                std_mean.append((np.std(mask),np.mean(mask)))
            
            # 左下六边形
            if i<h-2 and j>1:
                mask = [img[i+2,j-2], img[i+2,j-1], img[i+1,j-2], img[i+1,j-1], img[i+1,j], img[i,j-1], img[i,j]]
                std_mean.append((np.std(mask),np.mean(mask)))
            
            # 选取方差做小的模板的均值作为改点像素值
            img_smooth[i,j] = sorted(std_mean, key=lambda std_mean:std_mean[0])[0][1]
    return img_smooth

            

if __name__ == "__main__":

    img = cv2.imread("luna.png", 0)
    # 添加椒盐噪声
    img_noise = sp_noise(img)
    cv2.imwrite("pb2_noise.png", img_noise)
    # 平滑:选择保边缘平滑法
    from time import time
    start = time()
    img_smooth = smooth(img_noise)
    end = time()
    print ("time cost:", end-start)
    cv2.imwrite("pb2_result.png", img_smooth)