import numpy as np
import skimage.io as skio
from scipy import signal

from src.models.default_images import DefaultImages

DEFAULT_KERNEL_SIZE = 30
DEFAULT_SIGMA = 0.73


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
    denominator = np.sqrt(np.sum(abs_mean_centered_first ** 2) * np.sum(abs_mean_centered_second ** 2))

    return numerator / denominator


def gaussian_smoothening(image, size=DEFAULT_KERNEL_SIZE, sigma=DEFAULT_SIGMA):
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
    amount of blurring applied to the image. Ideally between 0 < sigma <= 1
    :return:
    """
    kernel = gaussian_kernel(size, sigma)
    return signal.convolve2d(image, kernel, mode='same', boundary='wrap')


def gaussian_kernel(size=DEFAULT_KERNEL_SIZE, sigma=DEFAULT_SIGMA):
    """
    Formulas:
    Gaussian Distribution Formula in 2D space - G(x, y) = (1 / 2*pi*std^2) * e^((x^2 + y^2) / 2*(std^2))
    Note: std = standard deviation. For our purposes, we will use an input sigma instead of standard deviation.
          In the method, we will subtract size // 2 from the coordinates to account for centering on pixel.

    :param size: The size of kernel to be used.
    :param sigma: The value of sigma determines the spread or width of the Gaussian curve, which in turn affects the
    amount of blurring applied to the image. Ideally between 0 < sigma <= 1
    :return:
    """
    denom = (1 / (2 * sigma ** 2))
    offset = size // 2

    kernel = np.fromfunction(
        lambda i, j: (1 / np.pi) * denom * np.exp(-((i - offset) ** 2 + (j - offset) ** 2) / denom),
        (size, size)
    )

    return kernel


def best_alignment_offset(aligning_image, base_image, window_width, window_height):
    target_x = int(base_image.shape[0] * 0.1)
    target_y = int(base_image.shape[1] * 0.1)
    target_window = base_image[target_x: target_x + window_width, target_y: target_y + window_height]

    best_matching_coord = (0, 0)
    best_diff = float("inf")

    for i in range(0, aligning_image.shape[0] - window_width):
        for j in range(0, aligning_image.shape[1] - window_height):
            current_window = aligning_image[i: i + window_width, j: j + window_height]
            diff = normalized_cross_correlation(current_window, target_window)
            if best_diff > diff:
                best_diff = diff
                best_matching_coord = (i, j)

    offset_x = best_matching_coord[0] - target_x
    offset_y = best_matching_coord[1] - target_y

    return offset_x, offset_y


class GlassPlateImage:
    def __init__(self, source_image):
        im_height = np.floor(source_image.shape[0] / 3.0).astype(int)
        self.source_im = source_image
        self.blue_channel_im = source_image[:im_height]
        self.green_channel_im = source_image[im_height: 2 * im_height]
        self.red_channel_im = source_image[2 * im_height:]

    def colorized(self):
        r = self.red_channel_im
        g = self.green_channel_im
        b = self.blue_channel_im

        # self.align(r, b, 100, 100)

        # align
        return np.dstack([r, g, b])

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

    blurred_im = gaussian_smoothening(gp_im.green_channel_im)
    skio.imshow(blurred_im)
    skio.show()
