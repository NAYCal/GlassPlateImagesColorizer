import numpy as np
import skimage.io as skio

from src.models.default_images import DefaultImages
from src.utils.mathematical_operations import canny_edge_detection, sum_of_squared_differences, \
    gaussian_smoothening_edge_subtraction, gaussian_smoothen
from src.utils.photo_adjustment_operations import crop_with_percent, pyramid_alignment

DEFAULT_ALIGNMENT_FN = pyramid_alignment
DEFAULT_EDGE_DETECTION_FUNC = gaussian_smoothening_edge_subtraction


class GlassPlateImage:
    def __init__(self, source_image, align_fn=DEFAULT_ALIGNMENT_FN,
                 edge_detection_fn=DEFAULT_EDGE_DETECTION_FUNC, to_edge=False, to_blurr=False):
        im_height = np.floor(source_image.shape[0] / 3.0).astype(int)
        self.source_im = source_image.copy()
        self.blue_channel_im = self.source_im[:im_height]
        self.green_channel_im = self.source_im[im_height: 2 * im_height]
        self.red_channel_im = self.source_im[2 * im_height: 3 * im_height]

        # Crop the photos to remove borders
        self.blue_channel_im = crop_with_percent(self.blue_channel_im)
        self.green_channel_im = crop_with_percent(self.green_channel_im)
        self.red_channel_im = crop_with_percent(self.red_channel_im)

        if to_blurr:
            self.blue_channel_im = gaussian_smoothen(self.blue_channel_im)
            self.green_channel_im = gaussian_smoothen(self.green_channel_im)
            self.red_channel_im = gaussian_smoothen(self.red_channel_im)

        self.align_fn = align_fn
        self.to_edge_fn = edge_detection_fn
        self.to_edge = to_edge

        self.colorized_im = None
        self.edged_colorized = None
        self.offsets = {}

        self.best_alignment_settings = {}

    def no_align_colorized(self):
        r = np.copy(self.red_channel_im)
        g = np.copy(self.green_channel_im)
        b = np.copy(self.blue_channel_im)

        return np.dstack([r, g, b])

    def colorized(self, base="g"):
        if self.colorized_im is not None:
            return self.colorized_im

        r, g, b = self.align(base)

        self.colorized_im = np.dstack([r, g, b])

        return self.colorized_im

    def aligned_edge_image(self, base="g"):
        r, g, b = self.align(base, True)
        return np.dstack([r, g, b])

    def edges_on_base_image(self, base="g"):
        r, g, b = self.align(base)
        r = r if base == "r" else self.to_edge_fn(r)
        g = g if base == "g" else self.to_edge_fn(g)
        b = b if base == "b" else self.to_edge_fn(b)

        return np.dstack([r, g, b])

    def align(self, base="g", result_is_edge=False):
        r = self.to_edge_fn(self.red_channel_im) if self.to_edge and base != "r" else self.red_channel_im
        g = self.to_edge_fn(self.green_channel_im) if self.to_edge and base != "g" else self.green_channel_im
        b = self.to_edge_fn(self.blue_channel_im) if self.to_edge and base != "b" else self.blue_channel_im

        clean_r = self.to_edge_fn(self.red_channel_im) if result_is_edge else self.red_channel_im
        clean_g = self.to_edge_fn(self.green_channel_im) if result_is_edge else self.green_channel_im
        clean_b = self.to_edge_fn(self.blue_channel_im) if result_is_edge else self.blue_channel_im

        if base == "r":
            r, self.offsets["r"] = clean_r, (0, 0)
            g, self.offsets["g"] = self.align_fn(g, r, clean_g)
            b, self.offsets["b"] = self.align_fn(b, r, clean_b)
        elif base == "g":
            r, self.offsets["r"] = self.align_fn(r, g, clean_r)
            g, self.offsets["g"] = clean_g, (0, 0)
            b, self.offsets["b"] = self.align_fn(b, g, clean_b)
        else:
            r, self.offsets["r"] = self.align_fn(r, b, clean_r)
            g, self.offsets["g"] = self.align_fn(g, b, clean_g)
            b, self.offsets["b"] = clean_b, (0, 0)

        return r, g, b

    # Find the alignment settings with the lowest diff
    # Measure with all the bases and edge detections
    def best_aligned(self):
        # Allows us to return back to original edge function
        this_edge_fn = self.to_edge_fn

        differences = []
        offsets = []
        settings = []
        images = []

        bases = ["r", "g", "b"]
        for base in bases:
            print("Base: ", base)
            no_edge_r, no_edge_g, no_edge_b = self.align(base, False)
            no_edge_diff = (sum_of_squared_differences(no_edge_r, no_edge_g) +
                            sum_of_squared_differences(no_edge_r, no_edge_b) +
                            sum_of_squared_differences(no_edge_b, no_edge_g))
            differences.append(no_edge_diff)
            offsets.append(self.offsets)
            settings.append((base, "No_Edge_Detection"))
            images.append(np.dstack([no_edge_r, no_edge_g, no_edge_b]))

            self.to_edge_fn = canny_edge_detection
            canny_r, canny_g, canny_b = self.align(base, False)
            canny_diff = (sum_of_squared_differences(canny_r, canny_g) +
                          sum_of_squared_differences(canny_r, canny_b) +
                          sum_of_squared_differences(canny_b, canny_g))
            differences.append(canny_diff)
            offsets.append(self.offsets)
            settings.append((base, "Canny_Edge_Detection"))
            images.append(np.dstack([canny_r, canny_g, canny_b]))

            self.to_edge_fn = gaussian_smoothening_edge_subtraction
            gauss_r, gauss_g, gauss_b = self.align(base, False)
            gauss_diff = (sum_of_squared_differences(gauss_r, gauss_g) +
                          sum_of_squared_differences(gauss_r, gauss_b) +
                          sum_of_squared_differences(gauss_b, gauss_g))
            differences.append(gauss_diff)
            offsets.append(self.offsets)
            settings.append((base, "Gauss_Edge_Detection"))
            images.append(np.dstack([gauss_r, gauss_g, gauss_b]))

        min_diff_index = differences.index(min(differences))
        self.best_alignment_settings = settings[min_diff_index]
        self.offsets = offsets[min_diff_index]
        self.colorized_im = images[min_diff_index]

        self.to_edge_fn = this_edge_fn
        return images[min_diff_index]

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
            self.colorized()
        skio.imshow(self.colorized_im)
        skio.show()


if __name__ == "__main__":
    im = DefaultImages.CHURCH.get_image()
    gp_im = GlassPlateImage(im, to_edge=False)

    skio.imshow(gp_im.best_aligned())
    skio.show()
