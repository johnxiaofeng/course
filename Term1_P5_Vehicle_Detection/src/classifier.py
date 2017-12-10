"""
module for car classifier
"""
import pickle
import time
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from feature_extractor import FeatureExtractor
from helpers import Helpers

class Classifier:
    """
    class for training, loading and saving car classifier
    """
    def __init__(self, color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat):
        """
        initialize the classifier with default parameter values
        """
        self.color_space = color_space
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat

        self.x_scaler = None
        self.svc = None

    def train(self):
        """
        train the support vector machine
        """
        start_time = time.time()
        x_train, x_test, y_train, y_test = self.prepare_training_data()
        end_time = time.time()
        print('Seconds to extract features from data is: {:.2f}'.format(end_time - start_time))

        start_time = time.time()
        self.svc = LinearSVC()
        self.svc.fit(x_train, y_train)
        end_time = time.time()
        print('Seconds to train SVC is: {:.2f}'.format(end_time - start_time))

        self.test(x_test, y_test)

    def test(self, x_test, y_test):
        """
        test the trained model on the test data
        """
        accuracy = self.svc.score(x_test, y_test)
        print('Test Accuracy of the SVC is: {:.4f}'.format(accuracy))

    def save(self, pickle_file_path):
        """
        save the trained svm to the pickle file with given path
        """
        with open(pickle_file_path, 'wb') as pickle_file:
            contents = {}
            contents['color_space'] = self.color_space
            contents['orient'] = self.orient
            contents['pix_per_cell'] = self.pix_per_cell
            contents['cell_per_block'] = self.cell_per_block
            contents['hog_channel'] = self.hog_channel
            contents['spatial_size'] = self.spatial_size
            contents['hist_bins'] = self.hist_bins
            contents['spatial_feat'] = self.spatial_feat
            contents['hist_feat'] = self.hist_feat
            contents['hog_feat'] = self.hog_feat
            contents['X_scalar'] = self.x_scaler
            contents['svc'] = self.svc
            pickle.dump(contents, pickle_file)

    def load(self, pickle_file_path):
        """
        load svm from the given pickle file
        """
        with open(pickle_file_path, 'rb') as pickle_file:
            contents = pickle.load(pickle_file)
            self.color_space = contents['color_space']
            self.orient = contents['orient']
            self.pix_per_cell = contents['pix_per_cell']
            self.cell_per_block = contents['cell_per_block']
            self.hog_channel = contents['hog_channel']
            self.spatial_size = contents['spatial_size']
            self.hist_bins = contents['hist_bins']
            self.spatial_feat = contents['spatial_feat']
            self.hist_feat = contents['hist_feat']
            self.hog_feat = contents['hog_feat']
            self.x_scaler = contents['X_scalar']
            self.svc = contents['svc']

    def prepare_training_data(self):
        """
        prepare the features to be used in training and testing
        """
        cars, notcars = Helpers.load_full_dataset()

        sample_size = 8790
        cars = cars[0:sample_size]
        notcars = notcars[0:sample_size]

        car_features = FeatureExtractor.get_features_from_pngs(
            cars,
            color_space=self.color_space,
            spatial_size=self.spatial_size,
            hist_bins=self.hist_bins,
            orient=self.orient,
            pix_per_cell=self.pix_per_cell,
            cell_per_block=self.cell_per_block,
            hog_channel=self.hog_channel,
            use_spatial_features=self.spatial_feat,
            use_hist_features=self.hist_feat,
            use_hog_features=self.hog_feat)

        notcar_features = FeatureExtractor.get_features_from_pngs(
            notcars,
            color_space=self.color_space,
            spatial_size=self.spatial_size,
            hist_bins=self.hist_bins,
            orient=self.orient,
            pix_per_cell=self.pix_per_cell,
            cell_per_block=self.cell_per_block,
            hog_channel=self.hog_channel,
            use_spatial_features=self.spatial_feat,
            use_hist_features=self.hist_feat,
            use_hog_features=self.hog_feat)

        features = np.vstack((car_features, notcar_features)).astype(np.float64)

        self.x_scaler = StandardScaler().fit(features)
        scaled_features = self.x_scaler.transform(features)
        labels = np.hstack((np.ones(len(cars)), np.zeros(len(notcars))))

        rand_state = np.random.randint(0, 100)
        x_train, x_test, y_train, y_test = train_test_split(
            scaled_features,
            labels,
            test_size=0.2,
            random_state=rand_state)

        print('Feature vector length:', len(x_train[0]))
        print('Training set size: {}, Test set size {}'.format(len(x_train), len(x_test)))
        return x_train, x_test, y_train, y_test

    def print_details(self):
        """
        print out the details of the trained classifier
        """
        print('------------------------------------')
        print('Car Classifier details: ')
        print('color_space: {}'.format(self.color_space))
        print('orient: {}'.format(self.orient))
        print('pix_per_cell: {}'.format(self.pix_per_cell))
        print('cell_per_block: {}'.format(self.cell_per_block))
        print('hog_channel: {}'.format(self.hog_channel))
        print('spatial_size: {}'.format(self.spatial_size))
        print('hist_bins: {}'.format(self.hist_bins))
        print('spatial_feat: {}'.format(self.spatial_feat))
        print('hist_feat: {}'.format(self.hist_feat))
        print('hog_feat: {}'.format(self.hog_feat))
        print('------------------------------------')


if __name__ == '__main__':
    classifier = Classifier(
        color_space='YCrCb',
        orient=9,
        pix_per_cell=8,
        cell_per_block=2,
        hog_channel='ALL',
        spatial_size=(32, 32),
        hist_bins=32,
        spatial_feat=True,
        hist_feat=True,
        hog_feat=True)
    classifier.print_details()
    classifier.train()

