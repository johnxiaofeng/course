"""
module for helper functions used
"""
import glob

import cv2
import numpy as np
import matplotlib.image as mpimg
from scipy.ndimage.measurements import label

from plot_helper import PlotHelper

class Helpers:
    """helper functions used in the project"""
    @staticmethod
    def draw_boxes(img, bboxes, color=(0, 0, 255), thickness=6):
        """
        create a copy of the image and draw the bboxes specified in the copied image
        img: input image
        bboxes: the list of bounding boxes
        color: the color for the edge of the bouding boxes
        thickness: the thickness of the edge of the bouding boxes
        """
        draw_img = np.copy(img)
        for vertex1, vertex2 in bboxes:
            cv2.rectangle(draw_img, vertex1, vertex2, color, thickness)
        return draw_img

    @staticmethod
    def load_full_dataset():
        """
        get image pathes of cars and notcars from the full vehicles and non-vehicles dataset
        """
        cars = Helpers.load_image_pathes('../data/vehicles/**/*.png')
        notcars = Helpers.load_image_pathes('../data/non-vehicles/**/*.png')
        print('Load {} car images, and {} notcar images'.format(len(cars), len(notcars)))
        return cars, notcars

    @staticmethod
    def load_small_dataset():
        """
        get image pathes of cars and notcars from the small vehicles and non-vehicles dataset
        """
        cars = Helpers.load_image_pathes('../data/vehicles_smallset/**/*.jpeg')
        notcars = Helpers.load_image_pathes('../data/non-vehicles_smallset/**/*.jpeg')
        print('Load {} car images, and {} notcar images'.format(len(cars), len(notcars)))
        return cars, notcars

    @staticmethod
    def get_heatmap(img, box_list, threshold=1, vis=False):
        """
        get a heat map from the image and the bounding box list
        """
        heat = np.zeros_like(img[:, :, 0]).astype(np.float)
        heat = Helpers.add_heat(heat, box_list)
        heat = Helpers.apply_threshold(heat, threshold)
        heatmap = np.clip(heat, 0, 255)

        labels = label(heatmap)
        draw_img = Helpers.draw_labeled_bboxes(np.copy(img), labels)
        if vis:
            PlotHelper.two_images(draw_img, 'Car Positions', heatmap, 'Heat Map', left_cmap=None, right_cmap='hot')

        return draw_img, heatmap

    @staticmethod
    def add_heat(heatmap, bbox_list):
        """
        add the heatness to the heatmap by counting the number of pixel occurance in bbox_list
        """
        for box in bbox_list:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        return heatmap

    @staticmethod
    def apply_threshold(heatmap, threshold):
        """
        apply the threshold to the heatmap to leave only necessary heat spots
        """
        heatmap[heatmap < threshold] = 0
        return heatmap

    @staticmethod
    def draw_labeled_bboxes(image, labels):
        """
        draw the labeled bboxes on the image
        """
        #iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # identify x and y values
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # draw the box on the image
            cv2.rectangle(image, bbox[0], bbox[1], (0, 0, 1), 6)
        return image

    @staticmethod
    def convert_color(image, color_space):
        """
        conver the color of the image to the give cspace
        images: should be in RGB color space
        """
        if color_space == 'LAB':
            colored_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        elif color_space == 'HSV':
            colored_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'HLS':
            colored_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            colored_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            colored_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        elif color_space == 'RGB':
            colored_image = np.copy(image)
        else:
            assert False
        return colored_image

    @staticmethod
    def load_image_pathes(image_path_pattern):
        """
        get all image pathes from the specified image path pattern
        """
        result = []
        images = glob.glob(image_path_pattern, recursive=True)
        for image in images:
            result.append(image)
        return result

    @staticmethod
    def load_image(image_path):
        """
        load the image from the given image path
        """
        return mpimg.imread(image_path)

    @staticmethod
    def load_jpeg(image_path):
        """
        load the jpeg image from the given image path
        since mpimg load jpeg into range (0, 255), and png to (0, 1)
        for consistency, convert jpeg also to (0, 1)
        """
        image = Helpers.load_image(image_path)
        image = image.astype(np.float32)/255
        return image

    @staticmethod
    def load_png(image_path):
        """
        load the png image from the given image path
        """
        return Helpers.load_image(image_path)
