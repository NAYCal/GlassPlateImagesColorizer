import numpy as np
import skimage.io as skio

from src.models.default_images import DefaultImages
from src.utils.photo_adjust_op import rolling_alignment, crop_with_percent

DEFAULT_ALIGNMENT_FUNC = rolling_alignment


class GlassPlateImage:
    def __init__(self, source_image, align_func=DEFAULT_ALIGNMENT_FUNC):
        im_height = np.floor(source_image.shape[0] / 3.0).astype(int)
        self.source_im = source_image.copy()
        self.blue_channel_im = self.source_im[:im_height]
        self.green_channel_im = self.source_im[im_height: 2 * im_height]
        self.red_channel_im = self.source_im[2 * im_height: 3 * im_height]

        # Crop the photos to remove borders
        self.blue_channel_im = crop_with_percent(self.blue_channel_im)
        self.green_channel_im = crop_with_percent(self.green_channel_im)
        self.red_channel_im = crop_with_percent(self.red_channel_im)

        self.align_func = align_func

        self.colorized_im = self.colorized()

    def colorized(self):
        r = self.red_channel_im
        g = self.green_channel_im
        b = self.blue_channel_im

        rb = self.align_func(r, b)
        gb = self.align_func(g, b)

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
        skio.imshow(self.colorized_im)
        skio.show()


if __name__ == "__main__":
    im = DefaultImages.CATHEDRAL.get_image()
    gp_im = GlassPlateImage(im)
    gp_im.show_colorized()
