"""
pipeline module for defining the whole work flow
"""

import cv2
import numpy as np

from classifier import Classifier
from feature_extractor import FeatureExtractor
from helpers import Helpers
from plot_helper import PlotHelper

from scipy.ndimage.measurements import label

class PipelineParameters:
    """
    define all the parameters used in the pipeline
    """
    # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    COLOR_SPACE = 'YCrCb'
    # HOG orientations
    ORIENT = 9
    # HOG pixels per cell
    PIX_PER_CELL = 8
    # HOG cells per block
    CELL_PER_BLOCK = 2
    # Can be 0, 1, 2, or "ALL"
    HOG_CHANNEL = 'ALL'
    # Spatial binning dimensions
    SPATIAL_SIZE = (32, 32)
    # Number of histogram bins
    HIST_BINS = 32
    # Spatial features on or off
    SPATIAL_FEATURE = True
    # Histogram features on or off
    HIST_FEATURE = True
    # HOG features on or off
    HOG_FEATURE = True

    # Threshold used in thresholding the heatmap of detected cars
    FRAME_HEAT_THRESHOLD = 2
    # Threshold uses in avaraged heatmap of detected cars
    AVG_HEAT_THRESHOLD = 2.8


class Pipeline:
    """
    class for the whole pipeling including traing the svm model and use it on image
    """
    def __init__(self, window_size):
        """
        initialize the pipeline, and create the classifier using the PipelineParameters
        """
        self.classifier = Classifier(
            color_space=PipelineParameters.COLOR_SPACE,
            orient=PipelineParameters.ORIENT,
            pix_per_cell=PipelineParameters.PIX_PER_CELL,
            cell_per_block=PipelineParameters.CELL_PER_BLOCK,
            hog_channel=PipelineParameters.HOG_CHANNEL,
            spatial_size=PipelineParameters.SPATIAL_SIZE,
            hist_bins=PipelineParameters.HIST_BINS,
            spatial_feat=PipelineParameters.SPATIAL_FEATURE,
            hist_feat=PipelineParameters.HIST_FEATURE,
            hog_feat=PipelineParameters.HOG_FEATURE)

        self.window_size = window_size
        self.heat_maps = []

    def train_model(self):
        """
        train the car classifier
        """
        self.classifier.print_details()
        self.classifier.train()

    def save_model(self, save_file_path):
        """
        save the model to the given save file path
        """
        self.classifier.save(save_file_path)

    def load_model(self, model_file_path):
        """
        load the model from the given save file path
        """
        self.classifier.load(model_file_path)
        self.classifier.print_details()
        self.check_model()

    def check_model(self):
        """
        check the parameter in the model, make sure it is the same as the pipeline
        """
        assert self.classifier.color_space == PipelineParameters.COLOR_SPACE
        assert self.classifier.orient == PipelineParameters.ORIENT
        assert self.classifier.pix_per_cell == PipelineParameters.PIX_PER_CELL
        assert self.classifier.cell_per_block == PipelineParameters.CELL_PER_BLOCK
        assert self.classifier.hog_channel == PipelineParameters.HOG_CHANNEL
        assert self.classifier.spatial_size == PipelineParameters.SPATIAL_SIZE
        assert self.classifier.hist_bins == PipelineParameters.HIST_BINS
        assert self.classifier.spatial_feat == PipelineParameters.SPATIAL_FEATURE
        assert self.classifier.hist_feat == PipelineParameters.HIST_FEATURE
        assert self.classifier.hog_feat == PipelineParameters.HOG_FEATURE

    def process_video_image(self, image):
        """
        the image pixel from video is [0,255], convert it to [0, 1]
        """
        image = image.astype(np.float32)/255
        result = self.process(image).astype(np.float32) * 255
        return result

    def process(self, image, vis=False):
        """
        process the image
        """
        return self.scan_all_cars(image, self.classifier, vis=vis)

    @staticmethod
    def find_cars(image, classifier, ystart, ystop, scale=1.0):
        """
        scan the image within given ystart and ystop for car
        """
        img_tosearch = image[ystart:ystop, :, :]
        ctrans_tosearch = Helpers.convert_color(img_tosearch, color_space=classifier.color_space)
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # define blocks and steps as above
        nxblocks = (ch1.shape[1] // classifier.pix_per_cell) - classifier.cell_per_block + 1
        nyblocks = (ch1.shape[0] // classifier.pix_per_cell) - classifier.cell_per_block + 1

        # 64 was the original sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // classifier.pix_per_cell) - classifier.cell_per_block + 1

        # compute individual channel HOG features for the entire image
        hog1 = FeatureExtractor.get_hog_features(ch1, classifier.orient, classifier.pix_per_cell, classifier.cell_per_block, vis=False, feature_vec=False)
        hog2 = FeatureExtractor.get_hog_features(ch2, classifier.orient, classifier.pix_per_cell, classifier.cell_per_block, vis=False, feature_vec=False)
        hog3 = FeatureExtractor.get_hog_features(ch3, classifier.orient, classifier.pix_per_cell, classifier.cell_per_block, vis=False, feature_vec=False)

        bboxes = []
        max_x_step = nxblocks - nblocks_per_window
        max_y_step = nyblocks - nblocks_per_window

        for y_step in range(max_y_step):
            y_pos_cells = y_step
            x_step = 0
            while x_step < max_x_step:
                x_pos_cells = x_step

                # extract hog for this patch
                hog_feat1 = hog1[y_pos_cells:y_pos_cells+nblocks_per_window, x_pos_cells:x_pos_cells+nblocks_per_window].ravel()
                hog_feat2 = hog2[y_pos_cells:y_pos_cells+nblocks_per_window, x_pos_cells:x_pos_cells+nblocks_per_window].ravel()
                hog_feat3 = hog3[y_pos_cells:y_pos_cells+nblocks_per_window, x_pos_cells:x_pos_cells+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                x_left_pixels = x_pos_cells * classifier.pix_per_cell
                y_top_pixels = y_pos_cells * classifier.pix_per_cell

                # extract the image patch
                subimg = cv2.resize(ctrans_tosearch[y_top_pixels:y_top_pixels+window, x_left_pixels:x_left_pixels+window], (64, 64))

                # get color features
                spatial_features = FeatureExtractor.extract_spatial_features(subimg, size=classifier.spatial_size)
                hist_features = FeatureExtractor.extract_hist_features(subimg, nbins=classifier.hist_bins)

                # scale features and make a prediction
                test_features = classifier.x_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                test_prediction = classifier.svc.predict(test_features)
                if test_prediction == 1:
                    xbox_left = np.int(x_left_pixels*scale)
                    ytop_draw = np.int(y_top_pixels*scale)
                    win_draw = np.int(window*scale)
                    bbox = [(xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + ystart + win_draw)]
                    bboxes.append(bbox)
                    x_step += 1
                else:
                    x_step += 2

        return bboxes

    def draw_detections(self, image, start_y, end_y, bboxes, title):
        """
        draw the detected boxes to the image
        """
        if len(bboxes) == 0:
            return

        draw_image = np.copy(image)
        draw_image = Helpers.draw_boxes(draw_image, bboxes, color=(0, 0, 1), thickness=2)

        image_width = draw_image.shape[1]
        region_of_interest = [[(0, start_y), (image_width, end_y)]]
        draw_image = Helpers.draw_boxes(draw_image, region_of_interest, color=(0, 1, 0), thickness=3)

        first_box = bboxes[0]
        box_width = first_box[1][0] - first_box[0][0]
        box_height = first_box[1][1] - first_box[0][1]
        box_rect = [[(0, start_y), (box_width, start_y + box_height)]]
        draw_image = Helpers.draw_boxes(draw_image, box_rect, color=(1, 1, 1), thickness=3)

        PlotHelper.color_image(draw_image, title=title)

    def scan_all_cars(self, image, classifier, vis=False):
        """
        scan the whole imag for cars
        """
        small_car_start_y = 380
        small_car_end_y = 540
        small_car_bboxes = Pipeline.find_cars(image, classifier, small_car_start_y, small_car_end_y, 1.2)

        middle_car_start_y = 380
        middle_car_end_y = 580
        middle_car_bboxes = Pipeline.find_cars(image, classifier, middle_car_start_y, middle_car_end_y, 1.8)

        big_car_start_y = 380
        big_car_end_y = 640
        big_car_bboxes = Pipeline.find_cars(image, classifier, big_car_start_y, big_car_end_y, 2.4)

        bboxes = small_car_bboxes + middle_car_bboxes + big_car_bboxes

        heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        heat = Helpers.add_heat(heat, bboxes)
        heat = Helpers.apply_threshold(heat, PipelineParameters.FRAME_HEAT_THRESHOLD)
        heat = np.clip(heat, 0, 255)
        self.add_heatmap(heat)

        avg_heatmap = self.get_average_heatmap()

        labels = label(avg_heatmap)
        draw_img = Helpers.draw_labeled_bboxes(np.copy(image), labels)

        if vis:
            image = np.copy(image)
            small = Helpers.draw_boxes(np.copy(image), small_car_bboxes, color=(0, 0, 1), thickness=3)
            middle = Helpers.draw_boxes(np.copy(image), middle_car_bboxes, color=(0, 0, 1), thickness=3)
            big = Helpers.draw_boxes(np.copy(image), big_car_bboxes, color=(0, 0, 1), thickness=3)
            PlotHelper.three_color_images(small, 'scale=1.2', middle, 'scale=1.8', big, 'scale=2.4')
            PlotHelper.two_images(draw_img, 'Car Positions', heat, 'Heat Map', left_cmap=None, right_cmap='hot')

        return draw_img

    def add_heatmap(self, heatmap):
        """
        add the heat map to the internal buffer
        """
        self.heat_maps.append(heatmap)
        while len(self.heat_maps) > self.window_size:
            self.heat_maps.pop(0)

    def get_average_heatmap(self):
        """
        get average heatmap from internal buffered heatmaps
        """
        average_heatmap = np.average(self.heat_maps, axis=0)
        average_heatmap = Helpers.apply_threshold(average_heatmap, PipelineParameters.AVG_HEAT_THRESHOLD)
        return average_heatmap


def train_model():
    """
    function to train a model and save the model to file model.p
    """
    pipeline = Pipeline(1)
    pipeline.train_model()
    pipeline.save_model('model.p')


if __name__ == '__main__':
    #train_model()

    import sys
    assert len(sys.argv) == 2
    image_path = sys.argv[1]

    pipe = Pipeline(1)
    pipe.load_model('model.p')

    image_type = image_path.split('.')[-1]
    if image_type == 'jpg':
        img = Helpers.load_jpeg(image_path)
    else:
        img = Helpers.load_png(image_path)

    pipe.process(img, True)
