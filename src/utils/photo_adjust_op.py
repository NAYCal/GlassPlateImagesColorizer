import numpy as np
import skimage as sk

from src.utils.mathematical_operations import (
    gaussian_smoothening_for_edge,
    sum_of_squared_differences,
)

# Percent to crop off from image
DEFAULT_CROP_OFFSET_RATIO = 0.1
# Where should rolling alignment center on
DEFAULT_ROLLING_CENTER_HEIGHT = 0
DEFAULT_ROLLING_CENTER_WIDTH = 0
# How much should rolling alignment roll
DEFAULT_ABS_ROLLING_HEIGHT_RATIO = 15
DEFAULT_ABS_ROLLING_WIDTH_RATIO = 15
# What is the minimum image size for pyramid alignment
DEFAULT_MIN_SIZE_FOR_PYRAMID = 500
# How much pyramid alignment should scale images per recursion
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


def rolling_alignment(input_image, base_image, clean_image=None, **kwargs):
    adj_func = kwargs.get("adj_func", DEFAULT_ADJUSTMENT_FUNCTION)
    compare_func = kwargs.get("compare_func", DEFAULT_ALIGNMENT_METRIC_FUNCTION)
    blurr_func = kwargs.get("blurr_func", DEFAULT_BLURRING_FUNCTION)
    center_height = kwargs.get("center_height", DEFAULT_ROLLING_CENTER_HEIGHT)
    center_width = kwargs.get("center_width", DEFAULT_ROLLING_CENTER_WIDTH)
    rolling_height = kwargs.get(
        "rolling_height", input_image.shape[0] // DEFAULT_ABS_ROLLING_HEIGHT_RATIO
    )
    rolling_width = kwargs.get(
        "rolling_width", input_image.shape[1] // DEFAULT_ABS_ROLLING_WIDTH_RATIO
    )
    to_blurr = kwargs.get("to_blurr", True)

    if clean_image is None:
        clean_image = np.copy(input_image)

    image = np.copy(input_image)
    base = np.copy(base_image)

    if to_blurr:
        image = blurr_func(image)
        base = blurr_func(image)

    best_image = clean_image
    best_diff = float("inf")
    best_offsets = (0, 0)

    height_rolling_range = range(-rolling_height, rolling_height)
    width_rolling_range = range(-rolling_width, rolling_width)
    for dx in height_rolling_range:
        for dy in width_rolling_range:
            shifted_im = adj_func(image, dx + center_height, dy + center_width)
            shifted_clean_im = adj_func(
                clean_image, dx + center_height, dy + center_width
            )
            diff = compare_func(shifted_im, base)
            if best_diff > diff:
                best_diff = diff
                best_image = shifted_clean_im
                best_offsets = (dx + center_height, dy + center_width)

    print(base_image.shape, "- Best offset: ", best_offsets)
    return best_image, best_offsets


def pyramid_align(
    input_image,
    base_image,
    clean_image=None,
    scaling_factor=DEFAULT_PYRAMID_SCALING_FACTOR,
    **kwargs
):
    assert input_image.shape == base_image.shape
    min_size_pyramid = kwargs.get("min_size_pyramid", DEFAULT_MIN_SIZE_FOR_PYRAMID)

    # clean_image without blurring or scaling.
    if clean_image is None:
        clean_image = np.copy(input_image)

    # Simply do normal alignment if image size is small
    if (
        base_image.shape[0] < min_size_pyramid
        and base_image.shape[1] < min_size_pyramid
    ):
        return rolling_alignment(input_image, base_image, clean_image, **kwargs)

    scaled_input = sk.transform.rescale(input_image, scaling_factor)
    scaled_base = sk.transform.rescale(base_image, scaling_factor)
    _, estimate_offset = pyramid_align(
        scaled_input, scaled_base, clean_image, scaling_factor, **kwargs
    )
    kwargs["rolling_height"] = abs(estimate_offset[0])
    kwargs["rolling_width"] = abs(estimate_offset[1])
    kwargs["center_height"] = estimate_offset[0]
    kwargs["center_width"] = estimate_offset[1]
    kwargs["to_blurr"] = True

    best_image, best_offsets = rolling_alignment(
        input_image, base_image, clean_image, **kwargs
    )

    return best_image, best_offsets
