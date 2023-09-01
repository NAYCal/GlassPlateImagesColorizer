import numpy as np
import skimage.io as skio
import skimage.transform

from src.models.default_images import DefaultImages


def normalized_cross_correlation(first_image, second_image):
    first_image_mean = np.mean(first_image)
    second_image_mean = np.mean(second_image)

    abs_mean_centered_first = np.abs(first_image - first_image_mean)
    abs_mean_centered_second = np.abs(second_image - second_image_mean)

    numerator = np.sum(abs_mean_centered_first * abs_mean_centered_second)
    denominator = np.sqrt(np.sum(abs_mean_centered_first ** 2) * np.sum(abs_mean_centered_second ** 2))

    return numerator / denominator

def gaussian_smoothening(image):
    pass


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
    colorized = gp_im.colorized()
    # skio.imshow(colorized)
    # skio.show()
