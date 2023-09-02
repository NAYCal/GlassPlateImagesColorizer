import numpy as np
import skimage as sk

from src.utils.mathematical_operations import (
    gaussian_smoothening_for_edge,
    sum_of_squared_differences,
)

DEFAULT_CROP_OFFSET_RATIO = 0.1
DEFAULT_ABS_ROLLING_HEIGHT = 15
DEFAULT_ABS_ROLLING_WINDOW_WIDTH = 15
DEFAULT_MIN_SIZE_FOR_PYRAMID = 1000
DEFAULT_PYRAMID_SCALING_FACTOR = 0.5
DEFAULT_BLURRING_FUNCTION = gaussian_smoothening_for_edge
DEFAULT_ALIGNMENT_METRIC_FUNCTION = sum_of_squared_differences
DEFAULT_ADJUSTMENT_FUNCTION = lambda image, offset_x, offset_y: np.roll(
    image, (offset_x, offset_y), axis=(0, 1)
)


def crop_with_percent(image, percent=DEFAULT_CROP_OFFSET_RATIO):
    x_crop_start = int(image.shape[0] * percent)
    y_crop_start = int(image.shape[1] * percent)

    x_crop_end = image.shape[0] - x_crop_start
    y_crop_end = image.shape[1] - y_crop_start

    return image.copy()[x_crop_start:x_crop_end, y_crop_start:y_crop_end]


def rolling_alignment(input_image, base_image, **kwargs):
    adj_func = kwargs.get("adj_func", DEFAULT_ADJUSTMENT_FUNCTION)
    compare_func = kwargs.get("compare_func", DEFAULT_ALIGNMENT_METRIC_FUNCTION)
    blurr_func = kwargs.get("blurr_func", DEFAULT_BLURRING_FUNCTION)
    to_blurr = kwargs.get("to_blurr", True)
    clean_image = kwargs.get("clean_image", np.copy(input_image)).copy()

    image = np.copy(input_image)
    base = np.copy(base_image)

    if to_blurr:
        image = blurr_func(image)
        base = blurr_func(image)

    best_image = clean_image
    best_diff = float("inf")
    best_offsets = (0, 0)

    height_rolling_range = range(
        -DEFAULT_ABS_ROLLING_HEIGHT, DEFAULT_ABS_ROLLING_HEIGHT
    )
    width_rolling_range = range(
        -DEFAULT_ABS_ROLLING_WINDOW_WIDTH, DEFAULT_ABS_ROLLING_WINDOW_WIDTH
    )
    for dx in height_rolling_range:
        for dy in width_rolling_range:
            shifted_im = adj_func(image, dx, dy)
            shifted_clean_im = adj_func(clean_image, dx, dy)
            diff = compare_func(shifted_im, base)
            if best_diff > diff:
                best_diff = diff
                best_image = shifted_clean_im
                best_offsets = (dx, dy)

    print("Best offset: ", best_offsets)
    return best_image


def pyramid_align(
    first_image, second_image, scaling_factor=DEFAULT_PYRAMID_SCALING_FACTOR
):
    scaled_first = sk.transform.rescale(first_image, scaling_factor)
    scaled_second = sk.transform.rescale(second_image, scaling_factor)


