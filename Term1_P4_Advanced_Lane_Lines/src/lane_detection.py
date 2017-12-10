"""
modeul for lane detection from warpped image
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

class LaneDetector:
    """
    class for detect the lanes in a warped image
    """
    # choose the number of sliding windows
    NUM_WINDOWS = 9

    # set the width of the window +/- LaneDetector.MARGIN
    MARGIN = 100

    # set minimum number of pixels found to recenter window
    MIN_NUM_PIXELS = 50

    # define conversions in x and y from pixels space to meters
    Y_METER_PER_PIXEL = 30 / 720
    X_METER_PER_PIXEL = 3.7 / 700

    def fit_lanes_by_sliding_window(self, binary_warped, plot_result=True):
        """
        fit lanes by sliding window method
        """
        # create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

        # histogram along all the columns in the lower half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

        # find the peak of the left and right halves of the histogram
        # these will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # set hight of windows
        window_height = np.int(binary_warped.shape[0] / LaneDetector.NUM_WINDOWS)

        # indentify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # current positions to be updated for each winodw
        leftx_current = leftx_base
        rightx_current = rightx_base

        # create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # step through the windows one by one
        for window in range(LaneDetector.NUM_WINDOWS):
            # identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1)*window_height
            win_y_high = binary_warped.shape[0] - window * window_height

            win_xleft_low = leftx_current - LaneDetector.MARGIN
            win_xleft_high = leftx_current + LaneDetector.MARGIN

            win_xright_low = rightx_current - LaneDetector.MARGIN
            win_xright_high = rightx_current + LaneDetector.MARGIN

            # draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            # identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # if you found > MIN_NUM_PIXELS pixels, recenter next window on their mean position
            if len(good_left_inds) > LaneDetector.MIN_NUM_PIXELS:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > LaneDetector.MIN_NUM_PIXELS:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        if len(leftx) == 0 or len(lefty) == 0:
            return None, None

        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        if len(rightx) == 0 or len(righty) == 0:
            return None, None

        # fit a second order ppolynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        if plot_result:
            # generate x and y values for plotting
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
            left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

            # plot the result
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            plt.imshow(out_img)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.show()

        return left_fit, right_fit

    def fit_lanes_in_margin(self, binary_warped, left_fit, right_fit, plot_result=True):
        """
        fit the lane from the previous fit result by seaching only in the region next to the previous fit
        """
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # search only a MARGIN around the the previous line position
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - LaneDetector.MARGIN))
        & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + LaneDetector.MARGIN)))

        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - LaneDetector.MARGIN))
        & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + LaneDetector.MARGIN)))

        # extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        if len(leftx) == 0 or len(lefty) == 0:
            return None, None

        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        if len(rightx) == 0 or len(righty) == 0:
            return None, None

        # fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # plot the result
        if plot_result:
            self.plot_fit_lanes(binary_warped, left_fit, right_fit, left_lane_inds, right_lane_inds)

        return left_fit, right_fit

    def plot_fit_lanes(self, binary_warped, left_fit, right_fit, left_lane_inds, right_lane_inds):
        """
        plot the fitted left and right lane
        """
        # generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)

        # color in left and right line pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # generate a polygon to illustrate the serach window area
        # and recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - LaneDetector.MARGIN, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + LaneDetector.MARGIN, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - LaneDetector.MARGIN, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + LaneDetector.MARGIN, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # draw the line onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255,0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255,0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    @staticmethod
    def get_lane_curvature(lane_fit):
        """
        get the lane radius of curvature in meters from lane fit coefficients
        """
        # generate the points based on the lane_fit coeffs
        ploty = np.linspace(0, 720 - 1, 720)
        lane_fitx = lane_fit[0]*ploty**2 + lane_fit[1]*ploty + lane_fit[2]

        # fit new polynomials to x,y in world space
        lane_fit_in_meter = np.polyfit(ploty*LaneDetector.Y_METER_PER_PIXEL, lane_fitx*LaneDetector.X_METER_PER_PIXEL, 2)

        # calculate the new radius of curvature in meter
        y_eval = np.max(ploty)
        curverad_meter = ((1 + (2*lane_fit_in_meter[0]*y_eval*LaneDetector.Y_METER_PER_PIXEL + lane_fit_in_meter[1])**2)**1.5) / np.absolute(2*lane_fit_in_meter[0])
        return curverad_meter

    @staticmethod
    def get_car_distance_to_lane(left_fit, right_fit):
        """
        get car distnace to the center of lane in meters given the left and right fit coefficients
        The distance can be either positive or negative
        if the distance is positive, it means the car is to the right of the lane center
        if the distance is negative, it means the car is to the left of the lane center
        """
        image_width = 1280
        image_height = 720

        car_position_x = image_width / 2
        left_lane_x = left_fit[0] * image_height**2 + left_fit[1]* image_height + left_fit[2]
        right_lane_x = right_fit[0] * image_height**2 + right_fit[1] * image_height + right_fit[2]
        lane_center_x = (left_lane_x + right_lane_x) / 2

        distance_to_car = (car_position_x - lane_center_x) * LaneDetector.X_METER_PER_PIXEL
        return distance_to_car

    @staticmethod
    def get_lane_curvature_fake():
        # generate some fake data to represent lane-line pixels
        ploty = np.linspace(0, 719, num=720)
        quadratic_coeff = 3e-4

        # for each y position generate random x position within +/- 50 pixels
        # of the line base position in each case (x=200 for left and x=900 for right)
        leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) for y in ploty])
        rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) for y in ploty])

        # reverse to match top to bottom in y
        leftx = leftx[::-1]
        rightx = rightx[::-1]

        # fit a second order polynomial to pixel positions in each fake lane line
        left_fit = np.polyfit(ploty, leftx, 2)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fit = np.polyfit(ploty, rightx, 2)
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # plot up the fake data
        mark_size = 3
        plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
        plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
        plt.xlim(0, 1280)
        plt.ylim(0, 720)
        plt.plot(left_fitx, ploty, color='green', linewidth=3)
        plt.plot(right_fitx, ploty, color='green', linewidth=3)
        plt.gca().invert_yaxis()
        plt.show()

        # define y-value where we want radius of curvature
        # i'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1+ (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        print(left_curverad, right_curverad)

        # fit new polynomials to x,y in world space
        left_fit_meter = np.polyfit(ploty*Y_METER_PER_PIXEL, leftx*X_METER_PER_PIXEL, 2)
        right_fit_cr = np.polyfit(ploty*Y_METER_PER_PIXEL, rightx*X_METER_PER_PIXEL, 2)

        # calculate the new radius of curvature
        left_curverad = ((1 + (2*left_fit_meter[0]*y_eval*Y_METER_PER_PIXEL + left_fit_meter[1])**2)**1.5) / np.absolute(2*left_fit_meter[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*Y_METER_PER_PIXEL + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        # now the radius of curvature is in meters
        print(left_curverad, 'm', right_curverad, 'm')

if __name__ == '__main__':
    LaneDetector.get_lane_curvature2()