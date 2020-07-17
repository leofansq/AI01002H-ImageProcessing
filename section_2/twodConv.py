"""
@leofansq
https://github.com/leofansq
"""
import cv2
import numpy as np 

def twodConv(f, k, p='zero'):
    """
    2D Conv
    Parameters:
        f: the input image (Gray Scale)
        k: conv kernel
        p: padding mode, e.g. 'replicate' or 'zero'
    Return:
        the conv result with the same shape as the input
    """
    # Shape of image & kernel
    h_f, w_f = f.shape
    h_k, w_k = k.shape

    r_h = (h_k-1)//2
    r_w = (w_k-1)//2

    # Padding
    f_pad = np.zeros([h_f+2*r_h, w_f+2*r_w])
    f_pad[r_h:h_f+r_h, r_w:w_f+r_w] = f

    if p == 'replicate':
        for i in range(r_h):
            f_pad[i, :] = f_pad[r_h, :]
            f_pad[-1-i, :] = f_pad[h_f-r_h+1, :]
        for i in range(r_w):
            f_pad[:, i] = f_pad[:, r_w]
            f_pad[:, -1-i] = f_pad[:, w_f-r_w+1]
    elif p == 'zero': pass
    else:
        raise ValueError("The third parameter should be replicate or zero")

    # Conv
    f_res = np.zeros_like(f)
    k = np.rot90(k, 2)

    for i in range(r_h, h_f+r_h):
        for j in range(r_w, w_f+r_w):
            roi = f_pad[i-r_h:i+r_h+1, j-r_w:j+r_w+1]
            f_res[i-r_h][j-r_w] = np.sum(roi*k)
    
    return f_res.astype(np.uint8)


if __name__ == "__main__":
    img = np.array([[1,2,3,4],
           [5,6,7,8],
           [9,8,7,6]])
    
    k = np.array([[1,2,3],
         [-1,0,1],
         [2,1,2]])
    
    ours = twodConv(img, k, 'replicate')
    # ours = twodConv(img, k)

    from scipy import signal
    sci_res = signal.convolve2d(img, k, mode='same', boundary='symm')
    # sci_res = signal.convolve2d(img, k, mode='same', boundary='fill')

    print (ours)
    print (sci_res)