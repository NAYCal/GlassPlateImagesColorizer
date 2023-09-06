import numpy as np
import skimage.io as skio

from src.models.default_images import DefaultImages
from src.models.colorize_glassplate_im import GlassPlateImage
from src.utils.image_display_operations import display_images
from src.utils.mathematical_operations import canny_edge_detection, gaussian_smoothening_edge_subtraction


def save_image(image, name, image_type):
    # Scale image data to 0-255 range for storing
    scaled_image = (image.copy() * 255).astype(np.uint8)

    # save the image
    file_name = '../out/' + image_type + '/' + name + '.jpg'
    skio.imsave(file_name, scaled_image)


# Divide by stages to save all the progresses made
# Iterate through all the default images
for unprocessed_image in DefaultImages:
    image_name = unprocessed_image.name
    print("Processing " + image_name)
    unprocessed_image = unprocessed_image.get_image()

    processing_image = GlassPlateImage(unprocessed_image, edge_detection_fn=gaussian_smoothening_edge_subtraction)

    print("-> no_alignment_image")
    no_alignment_image = processing_image.no_align_colorized()
    print("-> colorized_image")
    colorized_image = processing_image.colorized()
    print("-> edge_image")
    edge_image = processing_image.aligned_edge_image()
    print("-> edge_overlay_image")
    edge_overlay_image = processing_image.edges_on_base_image()

    r_offset = str(processing_image.offsets["r"][0]) + "_" + str(processing_image.offsets["r"][1])
    g_offset = str(processing_image.offsets["g"][0]) + "_" + str(processing_image.offsets["g"][1])
    b_offset = str(processing_image.offsets["b"][0]) + "_" + str(processing_image.offsets["b"][1])

    print("-> Saving images")
    save_image(no_alignment_image, "unaligned_" + image_name, "naive_colorized_images")
    save_image(colorized_image, image_name + "_r" + r_offset + "_g" + g_offset + "_b" + b_offset, "colorized_images")
    save_image(edge_image, image_name + "_gaussian_smoothening_edge_subtraction", "edge_images")
    save_image(edge_overlay_image, image_name + "_gaussian_smoothening_edge_subtraction", "edge_overlay_images")

    print("-> aligned_gaussian_smoothening_edge_subtraction")
    processing_image = GlassPlateImage(unprocessed_image,
                                       edge_detection_fn=gaussian_smoothening_edge_subtraction,
                                       to_edge=True)
    colorized_image_with_edges = processing_image.colorized()

    r_offset = str(processing_image.offsets["r"][0]) + "_" + str(processing_image.offsets["r"][1])
    g_offset = str(processing_image.offsets["g"][0]) + "_" + str(processing_image.offsets["g"][1])
    b_offset = str(processing_image.offsets["b"][0]) + "_" + str(processing_image.offsets["b"][1])

    print("-> Saving image")
    save_image(colorized_image_with_edges,
               image_name + "_r" + r_offset + "_g" + g_offset + "_b" + b_offset + "_gaussian_smoothening_edge_subtraction",
               "colorized_images_by_edges")

    print("-> aligned_canny_edge_detection")
    processing_image = GlassPlateImage(unprocessed_image,
                                       edge_detection_fn=canny_edge_detection,
                                       to_edge=True)
    colorized_image_with_edges = processing_image.colorized()

    r_offset = str(processing_image.offsets["r"][0]) + "_" + str(processing_image.offsets["r"][1])
    g_offset = str(processing_image.offsets["g"][0]) + "_" + str(processing_image.offsets["g"][1])
    b_offset = str(processing_image.offsets["b"][0]) + "_" + str(processing_image.offsets["b"][1])

    print("-> Saving image")
    save_image(colorized_image_with_edges,
               image_name + "_r" + r_offset + "_g" + g_offset + "_b" + b_offset + "_canny_edge_detection",
               "colorized_images_by_edges")

print("Stage 1 complete: Got all images processed through normal means")

# Iterate through all the default images
for unprocessed_image in DefaultImages:
    image_name = unprocessed_image.name
    print("Processing " + image_name)
    unprocessed_image = unprocessed_image.get_image()

    processing_image = GlassPlateImage(unprocessed_image)
    best_image = processing_image.best_aligned()

    r_offset = str(processing_image.offsets["r"][0]) + "_" + str(processing_image.offsets["r"][1])
    g_offset = str(processing_image.offsets["g"][0]) + "_" + str(processing_image.offsets["g"][1])
    b_offset = str(processing_image.offsets["b"][0]) + "_" + str(processing_image.offsets["b"][1])

    best_settings = str(processing_image.best_alignment_settings[0]) + "_" + str(processing_image.best_alignment_settings[1])

    print("-> Saving image")
    save_image(best_image,
               image_name + "_r" + r_offset + "_g" + g_offset + "_b" + b_offset + "_" + best_settings,
               "best_colorized_images")

print("Finish processing all images")
