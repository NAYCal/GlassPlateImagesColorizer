import numpy as np
import skimage.io as skio
from scipy import signal

from src.models.default_images import DefaultImages

DEFAULT_KERNEL_SIZE = 30
DEFAULT_SIGMA = 0.3
DEFAULT_CROP_OFFSET_RATIO = 0.15
DEFAULT_MIN_ALIGNMENT_WINDOW_HEIGHT = 50
DEFAULT_MIN_ALIGNMENT_WINDOW_WIDTH = 50


def alignment_offset(
    image, template, window_height=None, window_width=None, diff_func=None
):
    if window_height is None:
        window_height = DEFAULT_MIN_ALIGNMENT_WINDOW_HEIGHT
    if window_width is None:
        window_width = DEFAULT_MIN_ALIGNMENT_WINDOW_WIDTH
    if diff_func is None:
        diff_func = sum_of_squared_differences

    t_height, t_width = template.shape
    target_x = int((t_height - window_height) // 2)
    target_y = int((t_width - window_width) // 2)
    target_window = template[
        target_x : target_x + window_height, target_y : target_y + window_width
    ]
    diffs = np.zeros((t_height - window_height + 1, t_width - window_width + 1))

    for i in range(t_height - window_height + 1):
        for j in range(t_width - window_width + 1):
            curr_patch = image[i : i + window_height, j : j + window_width]
            diff = diff_func(curr_patch, target_window)
            diffs[i, j] = diff

    best_x, best_y = np.unravel_index(np.argmin(diffs), diffs.shape)  # 101, 71

    return best_x - target_x, best_y - target_y


def sum_of_squared_differences(first_image, second_image):
    return np.sum((first_image - second_image) ** 2)


def normalized_cross_correlation(first_image, second_image):
    """
    NCC to retrieve the differences between 2 images

    :param first_image:
    :param second_image:
    :return:
    """
    first_image_mean = np.mean(first_image)
    second_image_mean = np.mean(second_image)

    abs_mean_centered_first = np.abs(first_image - first_image_mean)
    abs_mean_centered_second = np.abs(second_image - second_image_mean)

    numerator = np.sum(abs_mean_centered_first * abs_mean_centered_second)
    denominator = np.sqrt(
        np.sum(abs_mean_centered_first**2) * np.sum(abs_mean_centered_second**2)
    )

    return numerator / denominator


def gaussian_smoothening_get_edge(image, size=DEFAULT_KERNEL_SIZE, sigma=DEFAULT_SIGMA):
    """
    Applies Gaussian smoothening on the input image then subtract original image to contrast the edges.
    How blurring works:
    Basic concept - Blurring image is similar to 'moshing' the colors in the area together.
                    What this means, we can achieve blurring effect by averaging pixel values of a 'window' in the
                    image.
    Problems      - This naive approach of blurring may have issues preserving edges.
    Solutions     - Using weighted kernel such that the center pixel (aka the pixel we are trying to blur color) has
                    more weight.
    Optimization  - Using the Gaussian distribution formula, which has a higher distribution in the center allows us to
                    achieve the same effect.

    :param image:
    :param size: The size of kernel to be used.
    :param sigma: The value of sigma determines the spread or width of the Gaussian curve, which in turn affects the
    amount of blurring applied to the image.
    :return:
    """
    kernel = gaussian_kernel(size, sigma)
    return signal.convolve2d(image, kernel, mode="same", boundary="wrap") - image


def gaussian_kernel(size=DEFAULT_KERNEL_SIZE, sigma=DEFAULT_SIGMA):
    """
    Formulas:
    Gaussian Distribution Formula in 2D space - G(x, y) = (1 / 2*pi*std^2) * e^((x^2 + y^2) / 2*(std^2))
    Note: std = standard deviation. For our purposes, we will use an input sigma instead of standard deviation.
          In the method, we will subtract size // 2 from the coordinates to account for centering on pixel.

    :param size: The size of kernel to be used.
    :param sigma: The value of sigma determines the spread or width of the Gaussian curve, which in turn affects the
    amount of blurring applied to the image.
    :return:
    """
    denom = 1 / (2 * sigma**2)
    offset = size // 2

    kernel = np.fromfunction(
        lambda i, j: (1 / np.pi)
        * denom
        * np.exp(-((i - offset) ** 2 + (j - offset) ** 2) / denom),
        (size, size),
    )

    return kernel


def crop_with_percent(image, percent=DEFAULT_CROP_OFFSET_RATIO):
    x_crop_start = int(image.shape[0] * percent)
    y_crop_start = int(image.shape[1] * percent)

    x_crop_end = image.shape[0] - x_crop_start
    y_crop_end = image.shape[1] - y_crop_start

    return image[x_crop_start:x_crop_end, y_crop_start:y_crop_end]


class GlassPlateImage:
    def __init__(self, source_image):
        im_height = np.floor(source_image.shape[0] / 3.0).astype(int)
        self.source_im = source_image
        self.blue_channel_im = source_image[:im_height]
        self.green_channel_im = source_image[im_height : 2 * im_height]
        self.red_channel_im = source_image[2 * im_height : 3 * im_height]

        # Crop the photos to remove borders
        self.blue_channel_im = crop_with_percent(self.blue_channel_im)
        self.green_channel_im = crop_with_percent(self.green_channel_im)
        self.red_channel_im = crop_with_percent(self.red_channel_im)

    def colorized(self):
        r = self.red_channel_im
        g = self.green_channel_im
        b = self.blue_channel_im

        rb = self.align(r, b)
        gb = self.align(g, b)

        return np.dstack([rb, gb, b])

    def align(self, first_image, second_image):
        assert first_image.shape == second_image.shape

        # Blur the images for better alignment, using edges to find alignment positions instead of raw brightness
        blurred_first_image = gaussian_smoothening_get_edge(first_image)
        blurred_second_image = gaussian_smoothening_get_edge(second_image)

        offset_x, offset_y = alignment_offset(blurred_first_image, blurred_second_image)

        result_image = np.roll(first_image, (offset_x, offset_y), axis=(0, 1))
        print("Offset: ", offset_x, offset_y)

        return result_image

    def show_original(self):
        skio.imshow(self.source_im)
        skio.show()

    def show_blue_channel(self):
        skio.imshow(self.blue_channel_im)
        skio.show()

    def show_green_channel(self):
        skio.imshow(self.green_channel_im)
        skio.show()

    def show_red_channel(self):
        skio.imshow(self.red_channel_im)
        skio.show()


if __name__ == "__main__":
    im = DefaultImages.CATHEDRAL.get_image()
    gp_im = GlassPlateImage(im)

    colorized_im = gp_im.colorized()
    skio.imshow(colorized_im)

    # blurred_im = gaussian_smoothening(gp_im.blue_channel_im)
    # skio.imshow(blurred_im)

    # cropped_im = crop_with_percent(gp_im.blue_channel_im)
    # skio.imshow(cropped_im)

    skio.show()
