"""
module for perspective transform
"""

import cv2
import numpy as np

class PerspectiveTransformer:
    """
    class the perform the perspective transform
    """
    LEFT_MARGIN = 250
    IMAGE_WIDTH = 1280
    IMAGE_HEIGHT = 720

    # The source and destination rect area for the perspective transform
    # in order of [bottom_left, top_left, top_right, bottom_right]
    SRC_RECT = [
        (575, 464),
        (258, 682),
        (1049, 682),
        (707, 464)]

    DST_RECT = [
        (LEFT_MARGIN, 0),
        (LEFT_MARGIN, IMAGE_HEIGHT),
        (IMAGE_WIDTH - LEFT_MARGIN, IMAGE_HEIGHT),
        (IMAGE_WIDTH - LEFT_MARGIN, 0)]

    def __init__(self):
        """
        initialize the transform matrix and inverse transform matrix
        """
        src_points = np.array(PerspectiveTransformer.SRC_RECT, dtype=np.float32)
        dst_pints = np.array(PerspectiveTransformer.DST_RECT, dtype=np.float32)

        self.M = cv2.getPerspectiveTransform(src_points, dst_pints)
        self.Minv = cv2.getPerspectiveTransform(dst_pints, src_points)

    def transform(self, input):
        """
        perspective transform the input image
        """
        dsize = (input.shape[1], input.shape[0])
        transformed = cv2.warpPerspective(input, self.M, dsize)
        return transformed

    def inverseTransform(self, input):
        """
        inverse transform the input image
        """
        dsize = (input.shape[1], input.shape[0])
        inverse_transformed = cv2.warpPerspective(input, self.Minv, dsize)
        return inverse_transformed
