"""
module for feature extractor
"""
import cv2
import numpy as np

from skimage.feature import hog
from sklearn.preprocessing import StandardScaler

from helpers import Helpers

class FeatureExtractor:
    """
    group of functions on feature extraction
    """
    @staticmethod
    def get_color_features_from_pngs(image_pathes, cspace='RGB', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256)):
        """
        extract features from a list of images
        """
        features = []
        for img_file in image_pathes:
            image = Helpers.load_png(img_file)
            feature_image = Helpers.convert_color(image, cspace)

            spatial_color_feature = FeatureExtractor.extract_spatial_features(feature_image, size=spatial_size)
            color_hist_feature = FeatureExtractor.extract_hist_features(feature_image, nbins=hist_bins, bins_range=hist_range)
            feature = np.concatenate((spatial_color_feature, color_hist_feature), axis=0)

            features.append(feature)
        return features

    @staticmethod
    def get_hog_features_from_pngs(image_pathes, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
        """
        extract the hog features from given list of image file pathes
        """
        features = []
        for file in image_pathes:
            image = Helpers.load_png(file)
            feature_image = Helpers.convert_color(image, cspace)

            hog_features = FeatureExtractor.extract_hog_features(
                image=feature_image,
                orient=orient,
                pix_per_cell=pix_per_cell,
                cell_per_block=cell_per_block,
                hog_channel=hog_channel)
            features.append(hog_features)
        return features

    @staticmethod
    def get_features_from_pngs(
            image_pathes, color_space='RGB', spatial_size=(32, 32), hist_bins=32,
            orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
            use_spatial_features=True, use_hist_features=True, use_hog_features=True):
        """
        extract spatial, hist and hog features from given list of images
        """
        features = []
        for file in image_pathes:
            image = Helpers.load_png(file)
            image_features = FeatureExtractor.extract_features(
                image=image,
                color_space=color_space,
                spatial_size=spatial_size,
                hist_bins=hist_bins,
                orient=orient,
                pix_per_cell=pix_per_cell,
                cell_per_block=cell_per_block,
                hog_channel=hog_channel,
                use_spatial_features=use_spatial_features,
                use_hist_features=use_hist_features,
                use_hog_features=use_hog_features)
            features.append(image_features)
        return features

    @staticmethod
    def extract_spatial_features(image, size=(32, 32)):
        """
        resize the img to the given size and flatten the image to a vector
        """
        features = cv2.resize(image, size).ravel()
        return features

    @staticmethod
    def extract_hist_features(image, nbins=32, bins_range=(0, 256)):
        """
        get the color histogram of the input img in all 3 channels
        and concatenate them together to get the final color histogram
        """
        assert image.shape[2] == 3

        channel1_hist = np.histogram(image[:, :, 0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(image[:, :, 1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(image[:, :, 2], bins=nbins, range=bins_range)
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        return hist_features

    @staticmethod
    def extract_hog_features(image, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
        """
        extract hog features from given image with given parameters
        hog_channel: 'ALL' will combine hog features from all channels, with 0, 1, 2, it only uses single channel
        """
        # if all channels are selected, get hog features for each channel and combine them
        if hog_channel == 'ALL':
            hog_features = []
            num_channels = image.shape[2]
            for channel_idx in range(num_channels):
                channel = image[:, :, channel_idx]
                channel_features = FeatureExtractor.get_hog_features(channel, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
                hog_features.append(channel_features)
            hog_features = np.ravel(hog_features)
        # if only one channel is seleted, get hog features only for that channel
        else:
            channel = image[:, :, hog_channel]
            hog_features = FeatureExtractor.get_hog_features(channel, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        return hog_features

    @staticmethod
    def get_hog_features(image, orient, pixel_per_cell, cell_per_block, vis=False, feature_vec=True):
        """
        get hog features from the img
        img: has to be a single color channel image or grayscaled image
        """
        if vis:
            features, hog_image = hog(
                image,
                orientations=orient,
                pixels_per_cell=(pixel_per_cell, pixel_per_cell),
                cells_per_block=(cell_per_block, cell_per_block),
                transform_sqrt=True,
                visualise=vis,
                feature_vector=feature_vec)
            return features, hog_image
        else:
            features = hog(
                image,
                orientations=orient,
                pixels_per_cell=(pixel_per_cell, pixel_per_cell),
                cells_per_block=(cell_per_block, cell_per_block),
                transform_sqrt=True,
                visualise=vis,
                feature_vector=feature_vec)
            return features

    @staticmethod
    def normalize_features(features):
        """
        normalize the features using StandardScaler
        """
        # fit a per-column scaler
        scaler = StandardScaler().fit(features)
        # apply the scaler to features
        normalized_features = scaler.transform(features)
        # return the normalized features
        return normalized_features

    @staticmethod
    def extract_features(
            image, color_space='RGB', spatial_size=(32, 32), hist_bins=32,
            orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
            use_spatial_features=True, use_hist_features=True, use_hog_features=True):
        """
        extract spatial, hist and hog features from given img with given parameters
        """
        #1) define an empty list to store image features
        image_features = []

        #2) apply color convertion if the color space is not RGB
        feature_image = Helpers.convert_color(image, color_space=color_space)

        #3) compute spatial features if flag is set
        if use_spatial_features:
            spatial_color_feature = FeatureExtractor.extract_spatial_features(feature_image, size=spatial_size)
            image_features.append(spatial_color_feature)

        #4) compute histogram features if flag is set
        if use_hist_features:
            color_hist_feature = FeatureExtractor.extract_hist_features(feature_image, nbins=hist_bins)
            image_features.append(color_hist_feature)

        #5) compute HOG features if flag is set
        if use_hog_features:
            hog_feature = FeatureExtractor.extract_hog_features(
                image=feature_image,
                orient=orient,
                pix_per_cell=pix_per_cell,
                cell_per_block=cell_per_block,
                hog_channel=hog_channel)
            image_features.append(hog_feature)

        #6) return concatenated array of features
        return np.concatenate(image_features)
