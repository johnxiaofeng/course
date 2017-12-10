"""
module for test the pipeline with video and images
"""
import glob

from helpers import Helpers
from pipeline import Pipeline
from moviepy.editor import VideoFileClip

import imageio
imageio.plugins.ffmpeg.download()

def process_video(input_file, output_file, process_function):
    """
    process the input video file, using the process function and output to the output vide file
    """
    #clip = VideoFileClip(input_file).subclip(40, 46)
    clip = VideoFileClip(input_file)

    #NOTE: this function expects color images!!
    white_clip = clip.fl_image(process_function)
    white_clip.write_videofile(output_file, audio=False)


def video_test(pipeline, input_file, output_file):
    """
    video test for the project video
    """
    process_video(input_file, output_file, pipeline.process_video_image)
    print('Finished processing ', input_file)


def images_test(pipeline, image_files_pattern):
    """
    test the pipeline on the basic images
    """
    images = glob.glob(image_files_pattern)
    for _, filename in enumerate(images):
        img = Helpers.load_jpeg(filename)
        pipeline.process(img, True)


if __name__ == '__main__':
    pipe = Pipeline(10)
    pipe.load_model('model.p')

    #video_test(pipe, 'videos/test_video.mp4', 'videos/test_video_output.mp4')
    video_test(pipe, 'videos/project_video.mp4', 'videos/project_video_output.mp4')
    #video_test(pipe, 'videos/challenge_video.mp4', 'videos/challenge_video_out.mp4')
    #images_test(pipe, 'images/test3.jpg')
    #images_test(pipe, 'images/special2.jpg')
