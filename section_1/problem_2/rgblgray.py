"""
@leofansq
https://github.com/leofansq
"""
import numpy as np

def rgblgray(f, method='NTSC'):
    """
    Parameters:
        f: image(RGB)
        method: 'average' or 'NTSC'
    Return:
        image in grayscale
    """
    if method == 'average':
        img_gray = f[:,:,0]/3 + f[:,:,1]/3 + f[:,:,2]/3
    elif method == 'NTSC':
        img_gray = f[:,:,0]*0.2989 + f[:,:,1]*0.5870 + f[:,:,2]*0.1140
    else:
        raise ValueError("The third parameter should be average or NTSC")
    
    img_gray = img_gray.astype(np.uint8)
    return img_gray