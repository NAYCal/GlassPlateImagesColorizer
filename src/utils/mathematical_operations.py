import numpy as np
import skimage
from scipy import signal
from skimage import io, color, feature
from skimage.filters import threshold_otsu

DEFAULT_KERNEL_SIZE = 45
DEFAULT_SIGMA = 7


def gaussian_smoothening_edge_subtraction(in_image, size=DEFAULT_KERNEL_SIZE, sigma=DEFAULT_SIGMA):
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

    :param in_image:
    :param size: The size of kernel to be used.
    :param sigma: The value of sigma determines the spread or width of the Gaussian curve, which in turn affects the
    amount of blurring applied to the image.
    :return:
    """
    kernel = gaussian_kernel(size, sigma)

    # Normalize channel values to 0-1 range
    channel = (in_image - np.min(in_image)) / (np.max(in_image) - np.min(in_image))

    # Apply Gaussian smoothing
    blurred_channel = signal.convolve2d(channel, kernel, mode="same", boundary="wrap")

    # Normalize the blurred channel to 0-1 range
    blurred_channel = (blurred_channel - np.min(blurred_channel)) / (
            np.max(blurred_channel) - np.min(blurred_channel)
    )

    # Subtract original channel to get edges
    edges = blurred_channel - channel

    # Compute Otsu's threshold
    otsu_threshold = threshold_otsu(edges)
    edges[edges < otsu_threshold] = 0
    edges[edges >= otsu_threshold] = 1

    return edges


def canny_edge_detection(channel, size=DEFAULT_KERNEL_SIZE, sigma=DEFAULT_SIGMA,
                         low_threshold_to_median = 0.01, high_threshold_to_median = 0.03):
    # Apply Gaussian smoothing
    blurred_channel = gaussian_smoothen(channel, size, sigma)

    # Compute the median of the pixel intensities
    median_intensity = np.median(blurred_channel)

    # Set thresholds based on the median of the pixel intensities
    low_threshold = low_threshold_to_median * median_intensity
    high_threshold = high_threshold_to_median * median_intensity

    # Apply Canny edge detection using scikit-image's feature.canny
    edges = feature.canny(
        blurred_channel, low_threshold=low_threshold, high_threshold=high_threshold
    )

    return edges


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
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2))
                     * np.exp(-((x - (size // 2)) ** 2 + (y - (size // 2)) ** 2) / (2 * sigma ** 2)),
        (size, size),
    )
    return kernel / np.sum(kernel)


def gaussian_smoothen(in_image, size=DEFAULT_KERNEL_SIZE, sigma=DEFAULT_SIGMA):
    kernel = gaussian_kernel(size, sigma)

    # Apply Gaussian smoothing
    blurred_image = signal.convolve2d(in_image, kernel, mode="same", boundary="wrap")

    return blurred_image


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
        np.sum(abs_mean_centered_first ** 2) * np.sum(abs_mean_centered_second ** 2)
    )

    return numerator / denominator
