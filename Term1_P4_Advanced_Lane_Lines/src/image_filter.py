"""
define the image filters used in pipeline
"""
from color_thresholds import *
from gradient_thresholds import *

def image_filter_basic(input):
    """
    image filter for the project video filtering in pipeline
    use a combination of sobel x, sobel y and magnitude for gradient filtering
    use a combination of hls s channel and hsv v channel for color filtering
    """
    gradient = gradient_threshold(input, sobelx_threshold=(12, 255), sobely_threshold=(25, 255), mag_threshold=(100, 255))
    color = color_threshold(input, hls_s_threshold=(100, 255), hsv_v_threshold=(50, 255))
    result = np.zeros_like(input[:,:,0])
    result[(gradient == 1) | (color == 1)] = 255
    return result

def image_filter_challenge(input):
    """
    image filter for challenge video filtering in pipeline
    use a combination of lab b channel and luv l channel for color filtering only
    """
    b_channel = lab_b_channel_threshold(input, threshold=(139, 255))
    l_channel = luv_l_channel_threshold(input, threshold=(174, 255))
    combined = np.zeros_like(input[:,:,0])
    combined[(l_channel == 1) | (b_channel == 1)] = 255
    return combined
