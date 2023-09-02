import numpy as np
from scipy import signal

DEFAULT_KERNEL_SIZE = 5
DEFAULT_SIGMA = 1


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
        lambda x, y: (1 / (2 * np.pi * sigma**2))
        * np.exp(-((x - (size // 2)) ** 2 + (y - (size // 2)) ** 2) / (2 * sigma**2)),
        (size, size),
    )
    return kernel / np.sum(kernel)


def gaussian_smoothening_for_edge(
    in_image, size=DEFAULT_KERNEL_SIZE, sigma=DEFAULT_SIGMA
):
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
    image = np.copy(in_image)
    blurred_image = signal.convolve2d(image, kernel, mode="same", boundary="wrap")
    return blurred_image - image


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
