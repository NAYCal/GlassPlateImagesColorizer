import numpy as np
import skimage as sk

from src.utils.mathematical_operations import sum_of_squared_differences

# Functions used: crop_with_percent
# Percent to crop off from image
DEFAULT_CROP_OFFSET_RATIO = 0.1

# Functions used: exhaustive_alignment, pyramid_alignment
# Measures how different two images are
DEFAULT_ALIGNMENT_METRIC_FUNCTION = sum_of_squared_differences
# Processes the image so that it matches the other
DEFAULT_IMAGE_PROCESS_FUNCTION = lambda image, offset_x, offset_y: np.roll(
    image, (offset_x, offset_y), axis=(0, 1)
)
# Determines what range of the image does exhaustive search checks
DEFAULT_EXHAUSTIVE_RANGE_X = 15
DEFAULT_EXHAUSTIVE_RANGE_Y = 15

# Functions used: pyramid_alignment
# Determines the minimum image size for pyramid alignment to be used
DEFAULT_MIN_SIZE_FOR_PYRAMID = 500


def crop_with_percent(image, percent=DEFAULT_CROP_OFFSET_RATIO):
    x_crop_start = int(image.shape[0] * percent)
    y_crop_start = int(image.shape[1] * percent)

    x_crop_end = image.shape[0] - x_crop_start
    y_crop_end = image.shape[1] - y_crop_start

    return image.copy()[x_crop_start:x_crop_end, y_crop_start:y_crop_end]


def exhaustive_alignment(
        input_image,
        base_image,
        clean_image=None,
        x_offset=0,
        y_offset=0,
        x_range=DEFAULT_EXHAUSTIVE_RANGE_X,
        y_range=DEFAULT_EXHAUSTIVE_RANGE_Y,
        process_fn=DEFAULT_IMAGE_PROCESS_FUNCTION,
        align_loss_fn=DEFAULT_ALIGNMENT_METRIC_FUNCTION,
):
    """
    Performs exhaustive search over 2 images, returning a modified input image with the best alignment.
    :param input_image:
    :param base_image:
    :param clean_image: A clean version of image in case the compared images are processed/blurred.
    :param x_offset:
    :param y_offset:
    :param x_range:
    :param y_range:
    :param process_fn:
    :param align_loss_fn:
    :return:
    """
    if clean_image is None:
        clean_image = np.copy(input_image)

    # Store results
    best_loss = float("inf")
    best_offsets = (0, 0)
    result_image = clean_image

    for i in range(-x_range, x_range):
        for j in range(-y_range, y_range):
            shifted_im = process_fn(input_image, i + x_offset, j + y_offset)
            clean_im = process_fn(clean_image, i + x_offset, j + y_offset)
            loss = align_loss_fn(shifted_im, base_image)

            if best_loss > loss:
                best_loss = loss
                best_offsets = (i + x_offset, j + y_offset)
                result_image = clean_im

    return result_image, best_offsets


def pyramid_alignment(
        input_image,
        base_image,
        clean_image=None,
        x_offset=0,
        y_offset=0,
        x_range=DEFAULT_EXHAUSTIVE_RANGE_X,
        y_range=DEFAULT_EXHAUSTIVE_RANGE_Y,
        process_fn=DEFAULT_IMAGE_PROCESS_FUNCTION,
        align_loss_fn=DEFAULT_ALIGNMENT_METRIC_FUNCTION,
        min_size_for_pyramid=DEFAULT_MIN_SIZE_FOR_PYRAMID,
):
    if clean_image is None:
        clean_image = np.copy(input_image)

    # Do a regular exhaustive alignment if the image is too small
    if (
            input_image.shape[0] < min_size_for_pyramid
            and input_image.shape[1] < min_size_for_pyramid
    ):
        return exhaustive_alignment(input_image, base_image, clean_image, x_offset, y_offset, x_range,
                                    y_range, process_fn, align_loss_fn)

    scaled_input = sk.transform.rescale(input_image, 0.5)
    scaled_base = sk.transform.rescale(base_image, 0.5)

    # Best estimated image and offset based on scaled down images
    # Note that the offsets of the scaled down images are different from the regular image
    _, estimate_offset = pyramid_alignment(scaled_input, scaled_base, clean_image, x_offset, y_offset,
                                           x_range, y_range, process_fn, align_loss_fn, min_size_for_pyramid)

    return exhaustive_alignment(input_image, base_image, clean_image, estimate_offset[0], estimate_offset[1],
                                estimate_offset[0], estimate_offset[1], process_fn, align_loss_fn)
