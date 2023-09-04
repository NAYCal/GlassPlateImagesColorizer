import numpy as np
import skimage.io as skio

from src.models.default_images import DefaultImages
from src.utils.mathematical_operations import gaussian_smoothen, canny_edge_detection, \
    gaussian_smoothening_edge_subtraction
from src.utils.photo_adjustment_operations import exhaustive_alignment, crop_with_percent, pyramid_alignment

DEFAULT_ALIGNMENT_FN = pyramid_alignment
DEFAULT_EDGE_DETECTION_FUNC = gaussian_smoothening_edge_subtraction


class GlassPlateImage:
    def __init__(self, source_image, align_fn=DEFAULT_ALIGNMENT_FN,
                 edge_detection_fn=DEFAULT_EDGE_DETECTION_FUNC, to_edge=False):
        im_height = np.floor(source_image.shape[0] / 3.0).astype(int)
        self.source_im = source_image.copy()
        self.blue_channel_im = self.source_im[:im_height]
        self.green_channel_im = self.source_im[im_height: 2 * im_height]
        self.red_channel_im = self.source_im[2 * im_height: 3 * im_height]

        # Crop the photos to remove borders
        self.blue_channel_im = crop_with_percent(self.blue_channel_im)
        self.green_channel_im = crop_with_percent(self.green_channel_im)
        self.red_channel_im = crop_with_percent(self.red_channel_im)

        self.align_fn = align_fn
        self.to_edge_fn = edge_detection_fn
        self.colorized_im = None
        self.to_edge = to_edge

    def colorized(self):
        r = np.copy(self.red_channel_im)
        g = np.copy(self.green_channel_im)
        b = np.copy(self.blue_channel_im)

        if self.to_edge:
            r = self.to_edge_fn(r)
            g = self.to_edge_fn(g)

        rb, _ = self.align_fn(r, b, np.copy(self.red_channel_im))
        gb, _ = self.align_fn(g, b, np.copy(self.green_channel_im))

        return np.dstack([rb, gb, b])

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

    def show_colorized(self):
        if self.colorized_im is None:
            self.colorized_im = self.colorized()
        skio.imshow(self.colorized_im)
        skio.show()


if __name__ == "__main__":
    im = DefaultImages.EMIR.get_image()
    gp_im = GlassPlateImage(im, to_edge=False)
    gp_im.show_colorized()

