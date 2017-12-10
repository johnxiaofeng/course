"""
helper functions for plotting images
"""
import matplotlib.pyplot as plt

class PlotHelper:
    @staticmethod
    def two_images(left_image, left_title, right_image, right_title, left_cmap=None, right_cmap='gray'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        fig.tight_layout()
        ax1.imshow(left_image, cmap=left_cmap)
        ax1.set_title(left_title, fontsize=50)
        ax2.imshow(right_image, cmap=right_cmap)
        ax2.set_title(right_title, fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    @staticmethod
    def three_gray_images(left_image, left_title, middle_image, middle_title, right_image, right_title):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
        fig.tight_layout()
        ax1.imshow(left_image, cmap='gray')
        ax1.set_title(left_title, fontsize=50)
        ax2.imshow(middle_image, cmap='gray')
        ax2.set_title(middle_title, fontsize=50)
        ax3.imshow(right_image, cmap='gray')
        ax3.set_title(right_title, fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    @staticmethod
    def color_image(image, title = None):
        """
        plot a color image
        """
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.show()

    @staticmethod
    def grayscale_image(image, title):
        """
        plot a gray scale image
        """
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.show()

    @staticmethod
    def grayscale_images(images, titles, num_columns=6):
        """
        plot a list of gray scale images together
        """
        assert len(images) == len(titles)

        num_images = len(images)
        num_rows = (num_images / num_columns) + 1
        plt.figure(1, figsize=(30, 30))
        for index in range(num_images):
            plt.subplot(num_rows, num_columns, index+1)
            plt.title(titles[index])
            plt.imshow(images[index], cmap='gray')
        plt.show()
