"""
@leofansq
https://github.com/leofansq
"""
import cv2
import numpy as np

def erode(img):
    """
    使用3*3矩形结构子 腐蚀操作
    Parameter:
        img: 待腐蚀图像
    Return:
        img_result: 腐蚀结果图像
    """
    # 初始化图像平移矩阵
    m_1 = np.array([[1,0,-1],[0,1,-1]], dtype=np.float32)
    m_2 = np.array([[1,0,0],[0,1,-1]], dtype=np.float32)
    m_3 = np.array([[1,0,1],[0,1,-1]], dtype=np.float32)
    m_4 = np.array([[1,0,-1],[0,1,0]], dtype=np.float32)
    m_5 = np.array([[1,0,1],[0,1,0]], dtype=np.float32)
    m_6 = np.array([[1,0,-1],[0,1,1]], dtype=np.float32)
    m_7 = np.array([[1,0,0],[0,1,1]], dtype=np.float32)
    m_8 = np.array([[1,0,1],[0,1,1]], dtype=np.float32)
    M = [m_1, m_2, m_3, m_4, m_5, m_6, m_7, m_8]

    # 9个平移后的图像取交集得到腐蚀结果
    img_result = img.copy()
    for i in M:
        img_shift = cv2.warpAffine(img, i, (img.shape[1],img.shape[0]))
        img_result = cv2.bitwise_and(img_result, img_shift)
    
    return img_result

def dilate(img):
    """
    使用3*3矩形结构子 膨胀操作
    Parameter:
        img: 待膨胀图像
    Return:
        img_result: 膨胀结果图像
    """
    # 初始化图像平移矩阵
    m_1 = np.array([[1,0,-1],[0,1,-1]], dtype=np.float32)
    m_2 = np.array([[1,0,0],[0,1,-1]], dtype=np.float32)
    m_3 = np.array([[1,0,1],[0,1,-1]], dtype=np.float32)
    m_4 = np.array([[1,0,-1],[0,1,0]], dtype=np.float32)
    m_5 = np.array([[1,0,1],[0,1,0]], dtype=np.float32)
    m_6 = np.array([[1,0,-1],[0,1,1]], dtype=np.float32)
    m_7 = np.array([[1,0,0],[0,1,1]], dtype=np.float32)
    m_8 = np.array([[1,0,1],[0,1,1]], dtype=np.float32)
    M = [m_1, m_2, m_3, m_4, m_5, m_6, m_7, m_8]

    # 9个平移后的图像取并集得到腐蚀结果
    img_result = img.copy()
    for i in M:
        img_shift = cv2.warpAffine(img, i, (img.shape[1],img.shape[0]))
        img_result = cv2.bitwise_or(img_result, img_shift)
    
    return img_result

def open_morph(img):
    """
    开运算
    Parameter:
        img: 待进行开运算的图像
    Return:
        img_result: 开运算结果图像
    """
    # 先腐蚀, 再膨胀
    img_result = erode(img)
    img_result = dilate(img_result)

    return img_result

def show_img(name, img):
    """
    Show the image

    Parameters:    
        name: name of window    
        img: image
    """
    cv2.namedWindow(name, 0)
    cv2.imshow(name, img)

# if __name__ == "__main__":
#     img = np.array([[0,0,0,0,0],[0,255,255,255,0],[255,255,255,255,255],[0,255,255,255,0],[0,0,0,0,0]], dtype=np.uint8)
#     show_img("origin", img)
#     img_erode = erode(img)
#     show_img("erode", img_erode)
#     img_dilate = dilate(img)
#     show_img("dilate", img_dilate)
#     img_open = open_morph(img)
#     show_img("open", img_open)

#     cv2.waitKey()
#     cv2.destroyAllWindows()
