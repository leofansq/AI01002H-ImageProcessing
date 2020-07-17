"""
@leofansq
https://github.com/leofansq
"""
import cv2
import numpy as np
import math

def gaussKernel(sig, m=None):
    """
    Generate a Gauss kernel.
    Parameters:
        sig: sigma
        m: the shape of the Gauss kernel is m*m
    Return:
        a Gauss kernel
    """
    # Cal & Judge m
    M = math.ceil(sig*3)*2 + 1
    if m:
        if m < M: 
            raise ValueError("m is smaller than it should be.")
        else: pass
    else:
        m = M
    
    # Generate kernel
    k = np.zeros((m,m))
    center = m//2
    s = sig**2

    for i in range(m):
        for j in range(m):
            x, y = i-center, j-center
            k[i][j] = (1/(2*math.pi*s)) * math.exp(-(x**2+y**2)/(2*s))

    k = k/np.sum(k)

    return k

if __name__ == "__main__":
    sig = 1
    m = None
    k = gaussKernel(sig, m)
    print (k)


    

