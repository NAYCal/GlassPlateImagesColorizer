import numpy as np
import skimage.io as skio

from src.models.default_images import DefaultImages
from src.models.colorize_glassplate_im import GlassPlateImage
from src.utils.image_display_operations import display_images


def save_image(image, image_name, image_type):
    # Scale image data to 0-255 range for storing
    scaled_image = (image.copy() * 255).astype(np.uint8)

    # save the image
    file_name = '../out/' + image_type + '/' + image_name + '.jpg'
    skio.imsave(file_name, scaled_image)


# Stores the image objects themselves
original_images = []
glass_plate_images = []
no_alignment_images = []
colorized_images = []
# colorized_edged_images = []

all_processed_images = []

#Stores the names
original_titles = []
no_alignment_titles = []
colorized_titles = []
# colorized_edged_titles = []

all_processed_titles = []

# Save the images
for val in DefaultImages:
    im_titles = val.name
    original_image = val.get_image()

    gpi_image = GlassPlateImage(original_image)
    no_alignment_image = gpi_image.no_align_colorized()
    colorized_image = gpi_image.colorized()
    # colorized_edged_image = gpi_image.colorized_edges()

    original_images.append(original_image)
    glass_plate_images.append(gpi_image)
    no_alignment_images.append(no_alignment_image)
    colorized_images.append(colorized_image)
    # colorized_edged_images.append(colorized_edged_image)

    all_processed_images.append([no_alignment_image, colorized_image])

    no_alignment_title = im_titles + "_not_aligned"
    colorized_title = im_titles + "_colorized_r_" + str(gpi_image.offsets["rb"]) + "_" + str(gpi_image.offsets["gb"])
    # colorized_edged_title = im_titles + "_edges"

    original_titles.append(im_titles)
    no_alignment_titles.append(no_alignment_title)
    colorized_titles.append(colorized_title)
    # colorized_edged_titles.append(colorized_edged_title)

    all_processed_titles.append([no_alignment_title, colorized_title])

    save_image(no_alignment_image, no_alignment_title, "no_alignment")
    save_image(colorized_image, colorized_title, "colorized_g_based")
    # save_image(colorized_edged_image, colorized_edged_title, "colorized_edge")

display_images(original_images, all_processed_images, "Colorized Prokudin-Gorskii glass plate images", original_titles, all_processed_titles)
