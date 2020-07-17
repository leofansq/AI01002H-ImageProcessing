"""
@leofansq
https://github.com/leofansq
"""
import cv2 

from rgblgray import rgblgray

def main(filename):
    """
    Main function
    """
    # Read image
    print ("Processing {} ...".format(filename))
    img = cv2.imread(filename)

    # BGR 2 RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR 2 RGB

    # Call function: rgblgray
    img_gray_a = rgblgray(img, 'average')
    img_gray_NTSC = rgblgray(img, 'NTSC')
    
    # Save image
    cv2.imwrite("{}_average.png".format(filename[:-4]), img_gray_a)
    cv2.imwrite("{}_NTSC.png".format(filename[:-4]), img_gray_NTSC)

if __name__ == "__main__":

    main("lena512color.tiff")
    main("mandril_color.tif")

