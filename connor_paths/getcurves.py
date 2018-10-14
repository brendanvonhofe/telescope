from scipy import misc
from skimage import measure, color
import matplotlib.pyplot as plt

fimg = misc.imread("elephant.png")

gimg = color.colorconv.rgb2grey(fimg)

contours = measure.find_contours(gimg, 0.8)

for n, contour in enumerate(contours):
    plt.plot(contour[:, 1], contour[:, 0], linewidth=2)

plt.show()
