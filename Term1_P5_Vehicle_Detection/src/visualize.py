"""
module for helping visualize different steps of the pipeline
"""

from feature_extractor import FeatureExtractor
from plot_helper import PlotHelper
from helpers import Helpers

def visualize_hog_features(image, color_space='YCrCb', orient=8, pixel_per_cell=8, cell_per_block=2):
    """visualize the hog features of the given image"""

    PlotHelper.color_image(image, 'car image')
    converted_img = Helpers.convert_color(image, color_space=color_space)

    channel1 = converted_img[:, :, 0]
    _, hog1 = FeatureExtractor.get_hog_features(
        image=channel1,
        orient=orient,
        pixel_per_cell=pixel_per_cell,
        cell_per_block=cell_per_block,
        vis=True,
        feature_vec=True
    )

    channel2 = converted_img[:, :, 1]
    _, hog2 = FeatureExtractor.get_hog_features(
        image=channel2,
        orient=orient,
        pixel_per_cell=pixel_per_cell,
        cell_per_block=cell_per_block,
        vis=True,
        feature_vec=True
    )

    channel3 = converted_img[:, :, 2]
    _, hog3 = FeatureExtractor.get_hog_features(
        image=channel3,
        orient=orient,
        pixel_per_cell=pixel_per_cell,
        cell_per_block=cell_per_block,
        vis=True,
        feature_vec=True
    )

    PlotHelper.grayscale_images(
        [channel1, hog1, channel2, hog2, channel3, hog3],
        ['channel1', 'channel1 hog', 'channel2', 'channel2 hog', 'channel3', 'channel3 hog'], 6)

if __name__ == '__main__':
    #image_path = '../data/vehicles/KITTI_extracted/17.png'
    image_path = '../data/non-vehicles/GTI/image5.png'

    input_image = Helpers.load_png(image_path)
    #visualize_hog_features(input_image, color_space='YCrCb', orient=16, pixel_per_cell=8, cell_per_block=2)
    visualize_hog_features(input_image, color_space='HSV', orient=16, pixel_per_cell=8, cell_per_block=2)
    #visualize_hog_features(input_image, color_space='RGB', orient=16, pixel_per_cell=8, cell_per_block=2)
