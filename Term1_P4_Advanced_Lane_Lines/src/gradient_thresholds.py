"""
functions for gradient thresholding
"""

import cv2
import numpy as np

def abs_sobel_threshold_gray(gray, orient='x', sobel_kernel=3, threshold=(20, 100)):
    """
    apply sobel threshold to the gray scale image
    """
    # Take the derivative in x or y given orient = 'x' or 'y'
    sobel = None
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Take the absolute value of the derivative or gradient
    sobel_abs = np.absolute(sobel)

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sobel_scaled = np.uint8(255 * sobel_abs / np.max(sobel_abs))

    # Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    sobel_binary = np.zeros_like(sobel_scaled)
    sobel_binary[(sobel_scaled >= threshold[0]) & (sobel_scaled <= threshold[1])] = 1

    # Return this mask as your binary_output image
    return sobel_binary


def abs_sobel_threshold(image, orient='x', sobel_kernel=3, threshold=(20, 100)):
    """
    applies Sobel x or y, it takes an absolute value and applies a threshold.
    image should be in RGB color space
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return abs_sobel_threshold_gray(gray, orient=orient, sobel_kernel=sobel_kernel, threshold=threshold)


def magnitude_threshold(image, sobel_kernel=3, threshold=(0, 255)):
    """
    applies Sobel x and y, and then computes the magnitude of the gradient and applies a threshold
    image should be in RGB color space
    """
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Calculate the magnitude
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    sobel_mag_scaled = np.uint8(255 * sobel_mag / np.max(sobel_mag))

    # 5) Create a binary mask where mag thresholds are met
    sobel_mag_binary = np.zeros_like(sobel_mag_scaled)
    sobel_mag_binary[(sobel_mag_scaled >= threshold[0]) & (sobel_mag_scaled <= threshold[1])] = 1

    # 6) Return this mask as your binary_output image
    return sobel_mag_binary


def gradient_direction_threshold(image, sobel_kernel=3, threshold=(0, np.pi/2)):
    """
    apply Sobel x and y, then computes the direction of the gradient and applies a threshold.
    image should be in RGB color space
    """
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the x and y gradients
    sobelx_abs = np.absolute(sobelx)
    sobely_abs = np.absolute(sobely)

    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    gradient = np.arctan2(sobely_abs, sobelx_abs)

    # 5) Create a binary mask where direction thresholds are met
    direction_image_binary = np.zeros_like(gradient, dtype=np.uint8)
    direction_image_binary[(gradient > threshold[0]) & (gradient < threshold[1])] = 1

    # 6) Return this mask as your binary_output image
    return direction_image_binary


def gradient_threshold(image, sobelx_threshold, sobely_threshold, mag_threshold):
    """
    gradient threshold with the combination of sobel x,y and magnitude
    """
    sobel_x = abs_sobel_threshold(image, orient='x', sobel_kernel=3, threshold=sobelx_threshold)
    sobel_y = abs_sobel_threshold(image, orient='y', sobel_kernel=3, threshold=sobely_threshold)
    magnitude = magnitude_threshold(image, sobel_kernel=3, threshold=(100, 255))

    result = np.zeros_like(sobel_x)
    result[(sobel_x == 1) & (sobel_y == 1) & (magnitude == 1)] = 1
    return result
