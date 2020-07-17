"""
@leofansq
https://github.com/leofansq
"""
import cv2
import matplotlib.pyplot as plt

from scanLine4e import scanLine4e

def main(filename):
    """
    Main function
    """
    # Read image
    print ("Processing {} ...".format(filename))
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # Call function :scanLine4e 
    mid_row = scanLine4e(img, img.shape[0]//2, 'row')
    mid_col = scanLine4e(img, img.shape[1]//2, 'column')

    # Plot
    fig = plt.figure()
    row = fig.add_subplot(2,1,1)
    column = fig.add_subplot(2,1,2)

    row.set_title("mid_row")
    column.set_title("mid_column")

    row.plot(mid_row)
    column.plot(mid_col)

    plt.tight_layout()
    plt.savefig("{}_result.png".format(filename[:-4]))



if __name__ == "__main__":

    main("cameraman.tif")
    main("einstein.tif")

