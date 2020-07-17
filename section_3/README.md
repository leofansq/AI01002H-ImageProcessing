# Section 3: 二维快速傅里叶变换

> 实验环境
> * Python 3.6.0
> * Opencv 3.1.0

## 问题1 图像二维快速傅里叶变换

### 问题描述
实现一个函数 F=dft2D(f), 其中 f 是一个灰度源图像,F 是其对应的二维快速傅里叶变换(FFT)图像. 具体实现要求按照课上的介绍通过两轮一维傅里叶变换实现。也就是首先计算源图像每一行的一维傅里叶变换,然后对于得到的结果计算其每一列的一维傅里叶变换。

### Code实现
实现思路如下：
* 复制图片对象，防止后续傅里叶操作影响原图，并将新对象转为复数格式
* 通过两轮不同维度的一维傅里叶变换实现二维快速傅里叶变换

```Python
def dft2D(f):
    """
    通过计算一维傅里叶变换实现图像二维快速傅里叶变换
    Parameters:
        f: image (Gray scale)
    Return:
        the result of FFT
    """
    F = f.copy()
    F = F.astype(np.complex128)

    # FFT for each row
    for i in range(F.shape[0]):
        F[i] = np.fft.fft(F[i])
    
    # FFT for each column
    for i in range(F.shape[1]):
        F[:, i] = np.fft.fft(F[:, i])
    
    return F
```

## 问题2 图像二维快速傅里叶逆变换

### 问题描述
实现一个函数 f=idft2D(F), 其中 F 是一个灰度图像的傅里叶变换,f 是其对应的二维快速傅里叶逆变换 (IFFT)图像,也就是灰度源图像. 具体实现要求按照课上的介绍通过类似正向变换的方式实现。

### Code实现
实现思路如下：
* 将傅里叶结果求共轭
* 对上步结果做二维快速傅里叶变换
* 对上一部结果除以MN，做共轭运算

```Python
def idft2D(F):
    """
    实现二维傅里叶逆变换
    Parameters:
        F : 灰度图像的傅里叶变换结果
    Return:
        傅里叶逆变换结果
    """
    # Conjugate
    f = F.conjugate()
    
    # 2D FFT
    for i in range(f.shape[0]):
        f[i] = np.fft.fft(f[i])
    for i in range(f.shape[1]):
        f[:, i] = np.fft.fft(f[:, i])
    
    # Divide by MN & Conjugate
    f = f/f.size
    f = np.abs(f.conjugate())

    return f
```

## 问题3 测试图像二维快速傅里叶变换与逆变换

### 问题描述
对于给定的输入图像 rose512.tif, 首先将其灰度范围通过归一化调整到[0,1]. 将此归一化的图像记为 f. 首先调用问题 1 下实现的函数 dft2D 计算其傅里叶变换,记为 F。然后调用问题 2 下的函数 idft2D 计算 F 的傅里叶逆变换,记为 g. 计算并显示误差图像 d = f-g.

### Code实现
实现思路如下：
* 读入图片并归一化
* 调用dft2D做二维快速傅里叶变换
* 调用idft2D做二维快速傅里叶逆变换
* 计算差值图像
* 绘制实验结果

运行方式：```python problem3.py```

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

from dft2D import dft2D
from idft2D import idft2D

def main(filepath):
    """
    Main function for problem 3.
    """
    # Read the image
    img = cv2.imread(filepath, 0)
    img = img/255.

    # 2D FFT
    img_fft = dft2D(img)

    # 2D IFFT 
    img_ifft = idft2D(img_fft)

    # Cal difference
    img_diff = img - img_ifft

    # Plot the result
    plt.subplot(221).set_title("Origin")
    plt.imshow(img, cmap='gray')

    plt.subplot(222).set_title("FFT")
    plt.imshow(np.log(np.abs(img_fft)+1), cmap='gray')

    plt.subplot(223).set_title("IFFT")
    plt.imshow(img_ifft, cmap='gray')

    plt.subplot(224).set_title("Difference")
    plt.imshow(np.round(img_diff), cmap='gray')

    plt.tight_layout()
    plt.savefig("problem3.png")

if __name__ == "__main__":
    main("rose512.tif")
```

### 实验结果与分析
从图1的实验结果图可以看出，原图与逆变换结果的差值图像为全黑，表明先后经过傅里叶变换和傅里叶逆变换两个过程不会对原图像产生影响。

<div align=center>
    <img src="./problem3.png">
    图1  问题三实验结果图
</div>

## 问题 4 计算图像的中心化二维快速傅里叶变换与谱图像

### 问题描述
我们的目标是复现下图中的结果。首先合成矩形物体图像,建议图像尺寸为 512×512,矩形位于图像中心,建议尺寸为 60 像素长,10 像素宽,灰度假设已归一化设为 1. 对于输入图像 f 计算其中心化二维傅里叶变换 F。然后计算对应的谱图像 S=log(1+abs(F)). 显示该谱图像。

### Code实现
实现思路如下：
* 根据要求合成底色为黑色的白色矩形物体图像，并归一化
* 为与中心化的二维傅里叶变换结果对比，对原图做二维傅里叶变换
* 计算原图的中心化二维傅里叶变换
* 计算中心化二维傅里叶变换的谱图像
* 绘制实验结果
  
运行方式：```python problem4.py```

```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

from dft2D import dft2D
from idft2D import idft2D

def main():
    """
    Main Function for problem 4.
    """
    # Generate the image
    img = np.zeros([512,512])
    img[226:285, 251:260] = 255
    img = img/255.

    # FFT
    img_fft = dft2D(img)

    # Centralization FFT
    img_fft_c = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_fft_c[i, j] = img_fft_c[i, j] * ((-1)**(i+j))

    img_fft_c = dft2D(img_fft_c)

    # Logarithmic transformation
    img_fft_clog = np.log(1 + np.abs(img_fft_c)) 

    # Plot
    plt.subplot(221).set_title("Origin")
    plt.imshow(img, cmap='gray')

    plt.subplot(222).set_title("FFT")
    plt.imshow(np.abs(img_fft), cmap='gray')

    plt.subplot(223).set_title("Centralization")
    plt.imshow(np.abs(img_fft_c), cmap='gray')

    plt.subplot(224).set_title("Log")
    plt.imshow(img_fft_clog, cmap='gray')

    plt.tight_layout()
    plt.savefig("problem4.png")

if __name__ == "__main__":
    main()
```

### 实验结果与分析
实验结果如图2所示。

<div align=center>
    <img src="./problem4.png">
    图2  问题四实验结果图——矩形物体图像的傅里叶变换
</div>