"""
functions to apply color thresholds
"""
import cv2
import numpy as np

def gray_threshold(image, threshold=(180, 255)):
    """
    convert image to grayscale and apply the threshold
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    binary = np.zeros_like(gray)
    binary[(gray > threshold[0]) & (gray <= threshold[1])] = 1
    return binary


def hls_l_channel_threshold(image, threshold=(220, 255)):
    """
    convert to HLS and get l channel, then apply color threshold to l channel
    image should be in RGB color space
    """
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    return apply_threshold(l_channel, threshold)


def hls_s_channel_threshold(image, threshold=(90, 255)):
    """
    convert to HLS and get S channel, then apply color threshold to s channel
    image should be in RGB color space
    """
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    return apply_threshold(s_channel, threshold)


def hls_h_channel_threshold(image, threshold=(15, 100)):
    """
    convert to HLS and get H channel, then apply color threshold to h channel
    image should be in RGB color space
    """
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    h_channel = hls[:, :, 0]
    return apply_threshold(h_channel, threshold)


def rgb_r_channel_threshold(image, threshold=(200, 255)):
    """
    apply threshold to r channel
    image should be in RGB color space
    """
    r_channel = image[:, :, 0]
    return apply_threshold(r_channel, threshold)


def hsv_v_channel_threshold(image, threshold=(50, 255)):
    """
    convert rgb image to hsv, then apply the color threshold to v channel
    image should be in RGB color space
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]
    return apply_threshold(v_channel, threshold)


def lab_b_channel_threshold(image, threshold=(50, 255)):
    """
    convert rgb image to lab, then apply the color threshold to b channel
    image should be in RGB color space
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    b_channel = lab[:, :, 2]
    return apply_threshold(b_channel, threshold)


def luv_l_channel_threshold(image, threshold=(225, 255)):
    """
    convert rgb image to luv, then apply the color threshold to l channel
    image should be in RGB color space
    """
    luv = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    l_channel = luv[:,:,0]
    return apply_threshold(l_channel, threshold)


def color_threshold(image, hls_s_threshold, hsv_v_threshold):
    """
    a color threshold with the combination of both hls_s channel and hsv_v channel
    image should be in RGB color space
    """
    s_channel = hls_s_channel_threshold(image, hls_s_threshold)
    v_channel = hsv_v_channel_threshold(image, hsv_v_threshold)

    binary = np.zeros_like(s_channel)
    binary[(s_channel == 1) & (v_channel == 1)] = 1
    return binary

def apply_threshold(channel, threshold):
    binary = np.zeros_like(channel)
    binary[(channel >= threshold[0]) & (channel <= threshold[1])] = 1
    return binary