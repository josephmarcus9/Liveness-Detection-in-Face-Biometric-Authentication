# import the necessary packages
from skimage import feature
import numpy as np
import numpy as np
from skimage import feature as skif
import cv2


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image):  # , eps=1e-7):
        lbp = skif.local_binary_pattern(image, self.numPoints, self.radius, method="nri_uniform")
        hist, _ = np.histogram(lbp, bins=np.arange(0, self.numPoints + 3),
                               range=(0, self.numPoints + 2))  # , normed=True)
        y_h = hist[:, :, 0]  # y channel
        cb_h = hist[:, :, 1]  # cb channel
        cr_h = hist[:, :, 2]  # cr
        hist_final = np.concatenate(y_h, cb_h, cr_h)
        return hist_final
