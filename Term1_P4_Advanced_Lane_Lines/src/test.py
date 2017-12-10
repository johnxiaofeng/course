"""
video test module for reading the test video, apply
the pipeline and output the result to a video file
"""
import imageio
imageio.plugins.ffmpeg.download()

from moviepy.editor import VideoFileClip
from pipeline import Pipeline
from image_filter import *

import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def process_video(input_file, output_file, process_function):
    """
    process the input video file, using the process function and output to the output vide file
    """
    #clip = VideoFileClip(input_file).subclip(1,5)
    clip = VideoFileClip(input_file)

    #NOTE: this function expects color images!!
    white_clip = clip.fl_image(process_function)
    white_clip.write_videofile(output_file, audio=False)


def process_first_frame(input_file, pipeline):
    """
    process the first frame from the video
    """
    first_frame = VideoFileClip(input_file).get_frame(0)
    pipeline.process_first_image(first_frame)


def video_test_basic():
    """
    video test for the project video
    """
    input_file = '../project_video.mp4'
    output_file = '../project_video_output.mp4'
    pipeline = Pipeline(image_filter_basic, 10)

    process_first_frame(input_file, pipeline)
    process_video(input_file, output_file, pipeline.process_subsequent_image)
    print('Finished processing ', input_file)


def vedio_test_challenge():
    """
    video test for the challenge video
    """
    input_file = '../challenge_video.mp4'
    output_file = '../challenge_video_output.mp4'
    pipeline = Pipeline(image_filter_challenge, 10)

    process_first_frame(input_file, pipeline)
    process_video(input_file, output_file, pipeline.process_subsequent_image)
    print('Finished processing ', input_file)


def images_test_basic():
    """
    test the pipeline on the basic images
    """
    images = glob.glob('../test_images/test*.jpg')
    pipeline = Pipeline(image_filter_basic, 1)
    for index, filename in enumerate(images):
        original = mpimg.imread(filename)
        result = pipeline.process_first_image(original)
        plt.imshow(result)
        plt.show()


def images_test_challenge():
    """
    test the pipeline on the challenge images
    """
    images = glob.glob('../test_images/challenge/challenge*.jpg')
    pipeline = Pipeline(image_filter_challenge, 1)
    for index, filename in enumerate(images):
        original = mpimg.imread(filename)
        result = pipeline.process_first_image(original)
        plt.imshow(result)
        plt.show()


if __name__ == '__main__':
    #images_test_basic()
    #images_test_challenge()

    video_test_basic()
    #vedio_test_challenge()