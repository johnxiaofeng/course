"""
"""
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from lane_detection import LaneDetector
from perspective_transform import PerspectiveTransformer
from image_undistortion import ImageUndistorter

class Line:
    """
    line class the keep the state of lane detection
    """
    MINIMUM_CURVATURE = 50

    def __init__(self, normalize_window_size):
        # set the normalize window size
        self.normalize_window_size = normalize_window_size
        # was the line detected in the last iteration
        self.detected = False
        # polynomial coefficients avaraged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the mose recent fit
        self.current_fit = []
        # radius of curvature
        self.radius_of_curvature = None

    def add_new_fit(self, lane_fit):
        if lane_fit is None:
            self.detected = False
            return False

        self.radius_of_curvature = LaneDetector.get_lane_curvature(lane_fit)
        if self.radius_of_curvature < Line.MINIMUM_CURVATURE:
            self.detected = False
            return False

        self.detected = True
        # add the lane_fit the current_fit list
        self.current_fit.append(lane_fit)
        while len(self.current_fit) > self.normalize_window_size:
            self.current_fit.pop(0)

        # update the best fit using the average of the fit results in the window
        self.best_fit = np.average(self.current_fit, axis=0)
        return True


class Pipeline:
    """
    the pipeline the process the image to get the binary image which contains the lane
    """
    def __init__(self, image_filter, normalize_window_size):
        """
        init the pipeline by passing in the image_filter
        """
        self.image_distorter = ImageUndistorter('camera_calibration.p')
        self.perspective_transformer = PerspectiveTransformer()
        self.lane_detector = LaneDetector()
        self.image_filter = image_filter
        self.left_lane = Line(normalize_window_size)
        self.right_lane = Line(normalize_window_size)

    def process_first_image(self, image):
        return self.process_image(image, True)

    def process_subsequent_image(self, image):
        if self.left_lane.detected and self.right_lane.detected:
            return self.process_image(image, False)
        else:
            return self.process_image(image, True)

    def process_image(self, image, use_sliding_window):
        """
        process each image and show the result of lane detection
        """
        undistorted = self.image_distorter.undistort_image(image)
        thresholded = self.image_filter(undistorted)
        transformed = self.perspective_transformer.transform(thresholded)

        if use_sliding_window:
             left_fit, right_fit = self.lane_detector.fit_lanes_by_sliding_window(transformed, False)
        else:
             left_fit, right_fit = self.lane_detector.fit_lanes_in_margin(transformed, self.left_lane.best_fit, self.right_lane.best_fit, False)

        self.update_lanes(left_fit, right_fit)

        result = self.blend_detection(undistorted, transformed, self.left_lane, self.right_lane)
        return result

    def update_lanes(self, left_fit, right_fit):
        left_succeed = self.left_lane.add_new_fit(left_fit)
        if left_succeed == False:
            print('invalid left lane detection')

        right_succeed = self.right_lane.add_new_fit(right_fit)
        if right_succeed == False:
            print('invalid right lane detection')

    def blend_detection(self, target, warped, left_lane, right_lane):
        """
        draw the result of lane detection for current image to the target image
        """
        # create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        left_fit = left_lane.best_fit
        right_fit = right_lane.best_fit

        # generate points from fitted polynomial
        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))

        # warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self.perspective_transformer.inverseTransform(color_warp)

        # combine the result with the original image
        result = cv2.addWeighted(target, 1, newwarp, 0.3, 0)

        # write car distance to center information to result
        distance_to_center = LaneDetector.get_car_distance_to_lane(left_fit, right_fit)
        if distance_to_center > 0: side = 'right'
        else: side = 'left'
        distance_text = 'Car is {:.3f}m to the {} of center '.format(abs(distance_to_center), side)
        cv2.putText(result, distance_text, (40, 80), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 1, cv2.LINE_AA)

        # write curvature information to result
        curvature_text = 'Lane Radius L: {:.3f}m, R: {:.3f}m, Avg: {:.3f}m'.format(
            left_lane.radius_of_curvature,
            right_lane.radius_of_curvature,
            (left_lane.radius_of_curvature + right_lane.radius_of_curvature)/2)
        cv2.putText(result, curvature_text, (40, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 1, cv2.LINE_AA)
        return result
