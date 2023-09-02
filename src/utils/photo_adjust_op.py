import numpy as np

from src.utils.mathematical_operations import (
    gaussian_smoothening_for_edge,
    sum_of_squared_differences,
)

DEFAULT_CROP_OFFSET_RATIO = 0.1
DEFAULT_ABS_ROLLING_HEIGHT = 15
DEFAULT_ABS_ROLLING_WINDOW_WIDTH = 15
DEFAULT_BLURRING_FUNCTION = gaussian_smoothening_for_edge
DEFAULT_ALIGNMENT_METRIC_FUNCTION = sum_of_squared_differences
DEFAULT_ADJUSTMENT_FUNCTION = lambda image, offset_x, offset_y: np.roll(
    image, (offset_x, offset_y), axis=(0, 1)
)


def rolling_alignment(
    first_in_image,
    second_in_image,
    adj_func=None,
    compare_func=None,
    blurr_func=None,
    to_blurr=True,
):
    if adj_func is None:
        adj_func = DEFAULT_ADJUSTMENT_FUNCTION
    if compare_func is None:
        compare_func = DEFAULT_ALIGNMENT_METRIC_FUNCTION
    if blurr_func is None:
        blurr_func = DEFAULT_BLURRING_FUNCTION

    first_image = np.copy(first_in_image)
    second_image = np.copy(second_in_image)

    if to_blurr:
        first_image = blurr_func(first_image)
        second_image = blurr_func(second_image)

    clean_image = np.copy(first_in_image)
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
            aligned_image = adj_func(first_image, dx, dy)
            aligned_clean_image = adj_func(clean_image, dx, dy)
            diff = compare_func(aligned_image, second_image)
            if best_diff > diff:
                best_diff = diff
                best_image = aligned_clean_image
                best_offsets = (dx, dy)

    print("Best offset: ", best_offsets)
    return best_image


def crop_with_percent(image, percent=DEFAULT_CROP_OFFSET_RATIO):
    x_crop_start = int(image.shape[0] * percent)
    y_crop_start = int(image.shape[1] * percent)

    x_crop_end = image.shape[0] - x_crop_start
    y_crop_end = image.shape[1] - y_crop_start

    return image.copy()[x_crop_start:x_crop_end, y_crop_start:y_crop_end]
