"""
module defining, training and saving the model
"""
import csv
import math
import os
import cv2

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

class Model:
    """class for training the model and saving the result"""
    def __init__(self):
        """init the model to the pilotnet"""
        self.model = CNNetwork.pilotnet()
        #self.model = CNNetwork.lenet()

    def train(self, csvfile_pathes, batch_size, epochs, output_file):
        """
        train the model with given csvfiles, batch_size and epochs
        csvfile_pathes: the list of csv file pathes containing the data collected from simulator
        batch_size: batch_size used in training and validation
        epochs: number of epochs to fit the model
        output_file: file path to save the result of the model
        """
        # load samples from all csvfiles passed in
        samples = DataLoader.load_samples_from_csvfiles(csvfile_pathes)

        # split all samples to training and validation
        train_samples, validation_samples = train_test_split(samples, test_size=0.2)

        # use the generator for both training and validation data
        train_generator = Generator.batch_generator(train_samples, batch_size)
        train_steps = math.ceil(len(train_samples) / batch_size)

        validation_generator = Generator.batch_generator(validation_samples, batch_size)
        validation_steps = math.ceil(len(validation_samples) / batch_size)

        history_object = self.model.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_steps,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            epochs=epochs,
            verbose=1)

        # save the result to the output_file
        self.model.save(output_file)

        # plot the training history for better understanding the learning process
        Model.plot_training_history(history_object)

    @staticmethod
    def plot_training_history(history_object):
        """
        draw the training history and plot the training loss and validation loss
        history_object: the history object returned from model.fit_generator
        """
        print(history_object.history.keys())
        plt.plot(history_object.history['loss'])
        plt.plot(history_object.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.xlabel('epoch')
        plt.ylabel('mean squared error loss')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.show()


class DataLoader:
    """helper functions to load data"""
    @staticmethod
    def load_samples_from_csvfiles(csvfile_pathes):
        """
        load samples from a list of csv files
        csvfile_pathes: the list of csv file pathes
        """
        samples = []
        for csvfile_path in csvfile_pathes:
            file_samples = DataLoader.load_samples_from_csvfile(csvfile_path)
            samples.extend(file_samples)
        return samples

    @staticmethod
    def load_samples_from_csvfile(csvfile_path):
        """
        load samples from a single csv file
        csvfile_path: the path to the csv file
        """
        samples = []
        csv_file_dir = os.path.dirname(csvfile_path)
        image_file_dir = csv_file_dir + '/IMG/'
        with open(csvfile_path) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                sample = DataLoader.load_sample_from_csvfile_line(line, image_file_dir)
                samples.append(sample)
        print('Loaded {} samples from {}'.format(len(samples), csvfile_path))
        return samples

    @staticmethod
    def load_sample_from_csvfile_line(line, directory):
        """
        get sample from a line in the csvfile
        line: a line from the csvfile
        directory: the directory path to the image folder
        """
        center_image = directory + os.path.basename(line[0])
        left_image = directory + os.path.basename(line[1])
        right_image = directory + os.path.basename(line[2])
        stearing = float(line[3])

        # not used data
        # throttle = float(line[4])
        # breaking = float(line[5])
        # speed = float(line[6])
        return [center_image, left_image, right_image, stearing]


class Generator:
    """define the generator used by model fit function"""
    @staticmethod
    def batch_generator(samples, batch_size):
        """
        define the data generator used in training. At each time, it will yield a dataset of batch_size
        batch_size: the batch size of samples at each yield
        """
        num_samples = len(samples)
        while True:
            shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]
                images, measurements = Generator.get_data_from_samples(batch_samples)
                x_train = np.array(images)
                y_train = np.array(measurements)
                yield shuffle(x_train, y_train)

    @staticmethod
    def get_data_from_samples(batch_samples):
        """
        get data from samples
        batch_samples: the list of samples for the batch
        """
        camera_correction = 0.2
        images = []
        measurements = []
        for center, left, right, measurement in batch_samples:
            # add center image
            center_image = cv2.imread(center)
            images.append(center_image)
            measurements.append(measurement)

            # add flipped center image
            center_image_flipped = np.fliplr(center_image)
            images.append(center_image_flipped)
            measurements.append(-measurement)

            # add left image
            left_measurement = measurement + camera_correction
            left_image = cv2.imread(left)
            images.append(left_image)
            measurements.append(left_measurement)

            # add flipped left image
            left_image_flipped = np.fliplr(left_image)
            images.append(left_image_flipped)
            measurements.append(-left_measurement)

            # add right image
            right_measurement = measurement - camera_correction
            right_image = cv2.imread(right)
            images.append(right_image)
            measurements.append(right_measurement)

            # add flipped right image
            right_image_flipped = np.fliplr(right_image)
            images.append(right_image_flipped)
            measurements.append(-right_measurement)
        return images, measurements


class CNNetwork:
    """convolution neural network used for model"""

    @staticmethod
    def pilotnet():
        """
        the pilot net from nvidia team
        https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
        """
        dropout_rate = 0.05
        model = Sequential()

        # corp non-necessary part of the image
        model.add(Cropping2D(cropping=((74, 20), (0, 0)), input_shape=(160, 320, 3)))

        # normalize the image
        model.add(Lambda(lambda x: (x / 255.0) - 0.5))

        # convolution 5x5: 3 x 66 x 320 -> 24 x 31 x 158
        model.add(Convolution2D(filters=24, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
        model.add(Dropout(dropout_rate))

        # convolution 5x5: 24 x 31 x 158 -> 36 x 14 x 77
        model.add(Convolution2D(filters=36, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
        model.add(Dropout(dropout_rate))

        # convolution 5x5: 36 x 14 x 77 -> 48 x 5 x 37
        model.add(Convolution2D(filters=48, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
        model.add(Dropout(dropout_rate))

        # convolution 3x3: 48 x 5 x 37 -> 64 x 3 x 35
        model.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
        model.add(Dropout(dropout_rate))

        # convolution 3x3: 64 x 3 x 35 -> 64 x 1 x 33
        model.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
        model.add(Dropout(dropout_rate))

        # a flatten layer
        model.add(Flatten())

        # a fully connected layer with 1164 outputs
        model.add(Dense(1164, activation='relu'))
        model.add(Dropout(dropout_rate))

        # a fully connected layer with 100 outputs
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(dropout_rate))

        # a fully connected layer with 50 outputs
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(dropout_rate))

        # a fully connected layer with 10 ouputs
        model.add(Dense(10, activation='relu'))
        model.add(Dropout(dropout_rate))

        # output layer
        model.add(Dense(1))

        model.compile(loss='mse', optimizer='adam')
        return model

    @staticmethod
    def lenet():
        """the simple lenet"""
        dropout_rate = 0.2
        model = Sequential()

        # crop out not important part of the image
        model.add(Cropping2D(cropping=((75, 20), (0, 0)), input_shape=(160, 320, 3)))

        # normalize the image
        model.add(Lambda(lambda x: (x / 255.0) - 0.5))

        model.add(Convolution2D(filters=6, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout_rate))

        model.add(Convolution2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout_rate))

        model.add(Flatten())

        model.add(Dense(120, activation='relu'))
        model.add(Dropout(dropout_rate))

        model.add(Dense(84, activation='relu'))
        model.add(Dropout(dropout_rate))

        model.add(Dense(1))

        model.compile(loss='mse', optimizer='adam')
        return model


if __name__ == '__main__':
    # parameter possible to set by command line parameters
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('batch_size', 64, "The batch size.")
    flags.DEFINE_string('output_file', 'default.h5', 'the output file for the model')
    flags.DEFINE_integer('epochs', 7, 'number of epochs')

    # build the model, train and save the result
    model = Model()

    # put all data set to be used for training in train_files
    train_files = [
        'data_collection/Data/counter_clockwise_3/driving_log.csv',
        'data_collection/Data/counter_clock/driving_log.csv',
        'data_collection/Data/counter_clockwise_return_to_center/driving_log.csv',
        'data_collection/Data/clockwise/driving_log.csv',
        'data_collection/Data/counter_clock2/driving_log.csv',
    ]
    model.train(train_files, FLAGS.batch_size, FLAGS.epochs, FLAGS.output_file)
