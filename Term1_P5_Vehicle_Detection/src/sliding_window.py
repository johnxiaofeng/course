import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from helpers import Helpers
from feature_extractor import FeatureExtractor

def slide_window(img, x_start_stop=(None, None), y_start_stop=(None, None), xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """
    function to apply sliding window to the input img
    x_start_stop defines the start and stop x position, if not defined, set to image width
    y_start_stop defines the start and stop y position, if not defined, set to image height
    xy_window is the size of the sliding window
    xy_overlap is the ration of overlapping between two consequential window in x and y
    """
    width = img.shape[1]
    height = img.shape[0]

    x_start = x_start_stop[0] if x_start_stop[0] is not None else 0
    x_stop = x_start_stop[0] if x_start_stop[0] is not None else width
    y_start = y_start_stop[0] if y_start_stop[0] is not None else 0
    y_stop = y_start_stop[1] if y_start_stop[1] is not None else height
    x_span = x_stop - x_start
    y_span = y_stop - y_start

    pixels_per_step_x = np.int(xy_window[0] * (1 - xy_overlap[0]))
    pixels_per_step_y = np.int(xy_window[1] * (1 - xy_overlap[1]))

    pixels_buffer_x = np.int(xy_window[0] * xy_overlap[0])
    pixels_buffer_y = np.int(xy_window[1] * xy_overlap[1])

    num_windows_x = np.int((x_span - pixels_buffer_x) / pixels_per_step_x)
    num_windows_y = np.int((y_span - pixels_buffer_y) / pixels_per_step_y)

    window_list = []
    for x in range(num_windows_x):
        for y in range(num_windows_y):
            top_left = (x * pixels_per_step_x + x_start, y * pixels_per_step_y + y_start)
            bottom_right = (top_left[0] + xy_window[0], top_left[1] + xy_window[1])
            window_list.append((top_left, bottom_right))
    return window_list


# defin a function you will pass an image and the list of windows to be searched
def search_windows(img, windows, clf, scaler, color_space='RGB', spatial_size=(32,32), hist_bins=32
                  , hist_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0
                  , use_spatial_features=True, use_hist_features=True, use_hog_features=True):
    #1) create an empty list to receive positive detection windows
    on_windows = []
    #2) iteratr over all windows in the list
    for window in windows:
        #3) extract the test window from the original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #4) extract features for that window
        features = single_img_features(test_img, color_space=color_space
                        , spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell
                        , cell_per_block=cell_per_block, hog_channel=hog_channel
                        , use_spatial_features=use_spatial_features, use_hist_features=use_hist_features, use_hog_features=use_hog_features)
        #5) scaler extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) predict using the classifier
        prediction = clf.predict(test_features)
        #7) if positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) return windows for positive detections
    return on_windows





if __name__ == '__main__':
    image = Helpers.load_image('images/bbox-example-image.jpg')

    windows = slide_window(
        image,
        x_start_stop=(None, None),
        y_start_stop=(None, None),
        xy_window=(128, 128),
        xy_overlap=(0.8, 0.8))

    window_image = Helpers.draw_boxes(image, windows, color=(0, 0, 255), thickness=2)
    plt.imshow(window_image)
    plt.show()
