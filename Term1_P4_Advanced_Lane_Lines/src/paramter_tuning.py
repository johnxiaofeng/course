"""
A helper Qt based tool for parameter tuning
"""
import cv2
import numpy as np
import matplotlib.image as mpimg

from PyQt5.QtCore import *
from PyQt5.QtGui import QImage, QPainter
from PyQt5.QtWidgets import QDialog, QApplication

from gradient_thresholds import *
from color_thresholds import *

class ParameterTuningDialog(QDialog):
    """
    The QDialog used to help parameter tuning
    """
    MAX_THRESHOLD = 255
    MIN_THRESHOLD = 0
    STEP = 1

    """
    Helper dialog for parameter tuning
    """
    def __init__(self, image_path, image_filter, parent=None):
        super(ParameterTuningDialog, self).__init__(parent)
        self.low_threshold = 0
        self.high_threshold = 255
        self.original_image = mpimg.imread(image_path)
        self.filter = image_filter
        self.qt_image = QImageHelper.get_qimage_from_rgb_image(self.original_image)

    def paintEvent(self, paint_event):
        if self.qt_image is not None:
            painter = QPainter()
            painter.begin(self)
            painter.drawImage(0, 0, self.qt_image)
            painter.end()

    def keyPressEvent(self, QKeyEvent):
        super(ParameterTuningDialog, self).keyPressEvent(QKeyEvent)
        if Qt.Key_Escape == QKeyEvent.key():
            app.exit(1)

        if Qt.Key_Up == QKeyEvent.key():
            self.increase_low_threshold()
        elif Qt.Key_Down == QKeyEvent.key():
            self.decrease_low_threshold()
        elif Qt.Key_Home == QKeyEvent.key():
            self.increase_high_threshold()
        elif Qt.Key_End == QKeyEvent.key():
            self.decrease_high_threshold()

        if Qt.Key_O == QKeyEvent.key():
            self.show_original_image()
        elif Qt.Key_B == QKeyEvent.key():
            self.update_filtered_image()
        elif Qt.Key_1 == QKeyEvent.key():
            self.show_rgb_r_channel()
        elif Qt.Key_2 == QKeyEvent.key():
            self.show_rgb_g_channel()
        elif Qt.Key_3 == QKeyEvent.key():
            self.show_rgb_b_channel()
        elif Qt.Key_4 == QKeyEvent.key():
            self.show_hls_h_channel()
        elif Qt.Key_5 == QKeyEvent.key():
            self.show_hls_l_channel()
        elif Qt.Key_6 == QKeyEvent.key():
            self.show_hls_s_channel()
        elif Qt.Key_7 == QKeyEvent.key():
            self.show_hsv_h_channel()
        elif Qt.Key_8 == QKeyEvent.key():
            self.show_hsv_s_channel()
        elif Qt.Key_9 == QKeyEvent.key():
            self.show_hsv_v_channel()
        elif Qt.Key_F1 == QKeyEvent.key():
            self.show_lab_l_channel()
        elif Qt.Key_F2 == QKeyEvent.key():
            self.show_lab_a_channel()
        elif Qt.Key_F3 == QKeyEvent.key():
            self.show_lab_b_channel()
        elif Qt.Key_F4 == QKeyEvent.key():
            self.show_luv_l_channel()

    def increase_high_threshold(self):
        self.high_threshold += ParameterTuningDialog.STEP
        if self.high_threshold >= ParameterTuningDialog.MAX_THRESHOLD:
            self.high_threshold = ParameterTuningDialog.MAX_THRESHOLD
        self.update_filtered_image()

    def decrease_high_threshold(self):
        self.high_threshold -= ParameterTuningDialog.STEP
        if self.high_threshold <= self.low_threshold:
            self.high_threshold = self.low_threshold
        self.update_filtered_image()

    def increase_low_threshold(self):
        self.low_threshold += ParameterTuningDialog.STEP
        if self.low_threshold >= self.high_threshold:
            self.low_threshold = self.high_threshold
        self.update_filtered_image()

    def decrease_low_threshold(self):
        self.low_threshold -= ParameterTuningDialog.STEP
        if self.low_threshold <= ParameterTuningDialog.MIN_THRESHOLD:
            self.low_threshold = ParameterTuningDialog.MIN_THRESHOLD
        self.update_filtered_image()

    def show_original_image(self):
        qimage = QImageHelper.get_qimage_from_rgb_image(self.original_image)
        self.update_qimage(qimage)

    def show_rgb_r_channel(self):
        r_channel = self.original_image[:,:,0]
        qimage = QImageHelper.get_qimage_from_gray_image(r_channel)
        self.update_qimage(qimage)
        print('RGB R Channel')

    def show_rgb_g_channel(self):
        g_channel = self.original_image[:,:,1]
        qimage = QImageHelper.get_qimage_from_gray_image(g_channel)
        self.update_qimage(qimage)
        print('RGB G Channel')

    def show_rgb_b_channel(self):
        b_channel = self.original_image[:,:,2]
        qimage = QImageHelper.get_qimage_from_gray_image(b_channel)
        self.update_qimage(qimage)
        print('RGB B Channel')

    def show_hls_h_channel(self):
        hls = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2HLS)
        h_channel = hls[:,:,0]
        qimage = QImageHelper.get_qimage_from_gray_image(h_channel)
        self.update_qimage(qimage)
        print('HLS H Channel')

    def show_hls_l_channel(self):
        hls = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2HLS)
        l_channel = hls[:,:,1]
        qimage = QImageHelper.get_qimage_from_gray_image(l_channel)
        self.update_qimage(qimage)
        print('HLS L Channel')

    def show_hls_s_channel(self):
        hls = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        qimage = QImageHelper.get_qimage_from_gray_image(s_channel)
        self.update_qimage(qimage)
        print('HLS S Channel')

    def show_hsv_h_channel(self):
        hsv = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2HSV)
        h_channel = hsv[:,:,0]
        qimage = QImageHelper.get_qimage_from_gray_image(h_channel)
        self.update_qimage(qimage)
        print('HSV H Channel')

    def show_hsv_s_channel(self):
        hsv = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2HSV)
        s_channel = hsv[:,:,1]
        qimage = QImageHelper.get_qimage_from_gray_image(s_channel)
        self.update_qimage(qimage)
        print('HSV S Channel')

    def show_hsv_v_channel(self):
        hsv = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2HSV)
        v_channel = hsv[:,:,2]
        qimage = QImageHelper.get_qimage_from_gray_image(v_channel)
        self.update_qimage(qimage)
        print('HSV V Channel')

    def show_luv_l_channel(self):
        luv = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2LUV)
        l_channel = luv[:,:,0]
        qimage = QImageHelper.get_qimage_from_gray_image(l_channel)
        self.update_qimage(qimage)
        print('LUV l Channel avg:{} max:{}'.format(np.average(l_channel), np.max(l_channel)))

    def show_lab_l_channel(self):
        lab = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:,:,0]
        qimage = QImageHelper.get_qimage_from_gray_image(l_channel)
        self.update_qimage(qimage)
        print('LAB L Channel')

    def show_lab_a_channel(self):
        lab = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2LAB)
        a_channel = lab[:,:,1]
        qimage = QImageHelper.get_qimage_from_gray_image(a_channel)
        self.update_qimage(qimage)
        print('LAB A Channel')

    def show_lab_b_channel(self):
        lab = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2LAB)
        b_channel = lab[:,:,2]
        qimage = QImageHelper.get_qimage_from_gray_image(b_channel)
        self.update_qimage(qimage)
        print('LAB B Channel avg B = {} {}'.format(np.average(b_channel), np.max(b_channel)))

    def update_qimage(self, image):
        self.qt_image = image
        self.update()

    def update_filtered_image(self):
        image = self.filter(self.original_image, self.low_threshold, self.high_threshold)
        qimage = QImageHelper.get_qimage_from_gray_image(image)
        self.update_qimage(qimage)
        print("Threshold {} ~ {}".format(self.low_threshold, self.high_threshold))


class QImageHelper:
    @staticmethod
    def get_qimage_from_rgb_image(original):
        image = np.copy(original)
        height, width, depth = original.shape
        bytes_per_line = width * depth
        return QImage(image, width, height, bytes_per_line, QImage.Format_RGB888)

    @staticmethod
    def get_qimage_from_gray_image(original):
        image = np.copy(original)
        height, width = image.shape
        bytes_per_line = width
        return QImage(image, width, height, bytes_per_line, QImage.Format_Grayscale8)


class Filters:
    @staticmethod
    def get_gray_image_from_binary_image(image):
        gray_image = np.zeros_like(image)
        gray_image[image == 1] = 255
        return gray_image

    @staticmethod
    def sobelx(input, low_threshold, high_threshold):
        image = abs_sobel_threshold(input, orient='x', threshold=(low_threshold, high_threshold))
        return Filters.get_gray_image_from_binary_image(image)

    @staticmethod
    def sobely(input, low_threshold, high_threshold):
        image = abs_sobel_threshold(input, orient='y', threshold=(low_threshold, high_threshold))
        return Filters.get_gray_image_from_binary_image(image)

    @staticmethod
    def gradient_mag(input, low_threshold, high_threshold):
        image = magnitude_threshold(input, threshold=(low_threshold, high_threshold))
        return Filters.get_gray_image_from_binary_image(image)

    @staticmethod
    def gradient_direction(input, low_threshold, high_threshold):
        low = (0.5 * np.pi) * (low_threshold / 255.0)
        high = (0.5 * np.pi) * (high_threshold / 255.0)
        print('Direction in raidus {} ~ {}'.format(low, high))
        image = gradient_direction_threshold(input, threshold=(low, high))
        return Filters.get_gray_image_from_binary_image(image)

    @staticmethod
    def hls_h(input, low_threshold, high_threshold):
        image = hls_h_channel_threshold(input, threshold=(low_threshold, high_threshold))
        return Filters.get_gray_image_from_binary_image(image)

    @staticmethod
    def hls_s(input, low_threshold, high_threshold):
        image = hls_s_channel_threshold(input, threshold=(low_threshold, high_threshold))
        return Filters.get_gray_image_from_binary_image(image)

    @staticmethod
    def rgb_r(input, low_threshold, high_threshold):
        image = rgb_r_channel_threshold(input, threshold=(low_threshold, high_threshold))
        return Filters.get_gray_image_from_binary_image(image)

    @staticmethod
    def find_lane_boundary(input, low_threshold, high_threshold):

        b_channel = lab_b_channel_threshold(input, threshold=(138, 255))
        combined = np.zeros_like(input[:,:,0])

        return Filters.get_gray_image_from_binary_image(combined)

if __name__=="__main__":
    import sys
    app = QApplication(sys.argv)
    print(sys.argv)

    assert len(sys.argv) == 2
    image_path = sys.argv[1]

    dialog = ParameterTuningDialog(image_path, Filters.find_lane_boundary)
    dialog.resize(1280, 720)
    dialog.show()
    app.exec_()