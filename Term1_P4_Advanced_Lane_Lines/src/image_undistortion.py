"""
ImageUndistorter
"""

import cv2
import matplotlib.image as mpimg
import pickle

from plot_helper import PlotHelper

class ImageUndistorter:
    """
    class to load camera calibration inforamtion from picle file
    add use the calibration result to do image undistortion.
    """
    def __init__(self, camera_calibration_pickle):
        """
        initialize the ImageUndistorter by loading camera calibration info from picle file
        """
        dist_pickle = pickle.load(open(camera_calibration_pickle, 'rb'))
        self.mtx = dist_pickle['mtx']
        self.dist = dist_pickle['dist']
        self.image_size = dist_pickle['image_size']

    def undistort_image(self, image):
        """
        perform undistortion to image
        """
        image_size = (image.shape[1], image.shape[0])
        assert((image_size[0] == self.image_size[0]) and (image_size[1] == self.image_size[1]))
        undistorted = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        return undistorted


if __name__ == '__main__':
    undistorter = ImageUndistorter('camera_calibration.p')
    original_image = mpimg.imread('../camera_cal/calibration1.jpg')
    result = undistorter.undistort_image(image)
    PlotHelper.two_images(image, 'original', result, 'undistorted')