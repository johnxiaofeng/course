"""
module for camera calibration
"""
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

class CameraCalibrator:
    """class for camera calibration"""
    def __init__(self, nx, ny):
        """
        init the calibrator by setting the nx and ny in the calibration board
        """
        self.nx = nx
        self.ny = ny
        self.mtx = None
        self.dist = None
        self.image_size = None

    def calibrate(self, calibration_image_path_pattern):
        """
        calibrate the camera by providing the calibration images pattern to be used in glob
        calibration_image_path_pattern: the path pattern of calibration images
        """
        # make a list of calibration images
        images = glob.glob(calibration_image_path_pattern)
        objpoints, imgpoints = self.get_reference_points(images, draw_corners=False)

        # get image size from the first calibration image
        first_image_path = images[0]
        self.image_size = CameraCalibrator.get_image_size(first_image_path)

        # calibrate the camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, self.image_size, None, None)
        self.mtx = mtx
        self.dist = dist

    def save(self, pickle_file_path):
        """
        save the calibration result to specified pickle_file_path
        """
        distortion_pickle = {}
        distortion_pickle['mtx'] = self.mtx
        distortion_pickle['dist'] = self.dist
        distortion_pickle['image_size'] = self.image_size
        pickle.dump(distortion_pickle, open(pickle_file_path, 'wb'))
        print('Calibration saved to {}'.format(pickle_file_path))

    def get_reference_points(self, image_paths_glob, draw_corners=False):
        """
        get reference points,return the object points and image points
        """
        # array to store object points and image points from all the images
        objpoints = []
        imgpoints = []

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(7,5,0)
        objp = np.zeros((self.ny * self.nx, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1,2)

        # iterate through all calibration images and collect object and image points
        for index, image_path in enumerate(image_paths_glob):
            print('get reference point from {}'.format(image_path))
            image = mpimg.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
                if draw_corners:
                    cv2.drawChessboardCorners(image, (self.nx, self.ny), corners, ret)
                    plt.imshow(image)
                    plt.show()

        return objpoints, imgpoints

    @staticmethod
    def get_image_size(image_path):
        image = mpimg.imread(image_path)
        return (image.shape[1], image.shape[0])


if __name__ == '__main__':
    calibrator = CameraCalibrator(9, 6)
    calibrator.calibrate('../camera_cal/calibration*.jpg')
    calibrator.save('camera_calibration.p')
