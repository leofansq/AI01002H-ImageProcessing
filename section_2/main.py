"""
@leofansq
https://github.com/leofansq
"""
import cv2
import numpy as np

from twodConv import twodConv
from gaussKernel import gaussKernel

def main(filename):
    """
    Main function for problem 3
    """
    # Read image
    print ("Processing {} ...".format(filename))
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # Generate kernel & Conv
    sig_list = [1,2,3,5]

    for sig in sig_list:
        # Generate Gauss kernel
        k = gaussKernel(sig)

        #Conv2D
        res = twodConv(img, k)
        cv2.imwrite("{}_{}.png".format(filename[:-4],sig), res)

        # Compare with opencv
        if sig == 1:
            res_cv2 = cv2.GaussianBlur(img, (7,7), 1, borderType=0)
            sub = res_cv2 - res
            cv2.imwrite("{}_sub.png".format(filename[:-4]), sub)
        
        # Padding mode: replicate VS zero
        rep_list = ['lena.png', 'mandril.png']
        if filename in rep_list:
            res_rep = twodConv(img, k, 'replicate')
            cv2.imwrite("{}_rep_{}.png".format(filename[:-4],sig), res_rep)
    


if __name__ == "__main__":
    img_list = ['cameraman.png', 'einstein.png', 'lena.png', 'mandril.png']
    for i in img_list:
        main(i)


