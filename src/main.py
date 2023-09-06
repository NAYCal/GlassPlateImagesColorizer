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


def process_everything():
    all_settings = ["colorized_images", "colorized_images_by_edges", "edge_images", "edge_overlay_images",
                    "naive_colorized_images", "gaussian_blurred_images"]
    all_colors = ["r", "g", "b"]
    for base in all_colors:
        print("Processing base: ", base)
        for process_type in all_settings:
            print(" -> Processing setting: ", process_type)
            process_all_images(process_type, base)

    print("Processing best images")
    process_all_images("best_colorized_images")


def process_all_images(process_type="colorized_images", base="g"):
    for unprocessed_image in DefaultImages:
        image_name = unprocessed_image.name
        print(" --> Processing " + image_name)
        unprocessed_image = unprocessed_image.get_image()

        process_image(unprocessed_image, image_name, process_type, base)


def process_image(original_image, image_name, process_type, base="g"):
    match process_type:
        case "colorized_images":
            gpi_image = GlassPlateImage(original_image, to_edge=False)
            processed_image = gpi_image.colorized(base)

            r_offset = str(gpi_image.offsets["r"][0]) + "_" + str(gpi_image.offsets["r"][1])
            g_offset = str(gpi_image.offsets["g"][0]) + "_" + str(gpi_image.offsets["g"][1])
            b_offset = str(gpi_image.offsets["b"][0]) + "_" + str(gpi_image.offsets["b"][1])

            file_name = image_name + "_r" + r_offset + "_g" + g_offset + "_b" + b_offset
            save_image(processed_image, file_name, process_type)
        case "colorized_images_by_edges":
            gpi_image = GlassPlateImage(original_image, edge_detection_fn=gaussian_smoothening_edge_subtraction,
                                        to_edge=True)
            processed_image = gpi_image.colorized(base)

            r_offset = str(gpi_image.offsets["r"][0]) + "_" + str(gpi_image.offsets["r"][1])
            g_offset = str(gpi_image.offsets["g"][0]) + "_" + str(gpi_image.offsets["g"][1])
            b_offset = str(gpi_image.offsets["b"][0]) + "_" + str(gpi_image.offsets["b"][1])

            file_name = image_name + "_r" + r_offset + "_g" + g_offset + "_b" + b_offset + "_gaussian_smoothening_edge_subtraction"
            save_image(processed_image, file_name, process_type)

            gpi_image = GlassPlateImage(original_image, edge_detection_fn=canny_edge_detection, to_edge=True)
            processed_image = gpi_image.colorized(base)

            r_offset = str(gpi_image.offsets["r"][0]) + "_" + str(gpi_image.offsets["r"][1])
            g_offset = str(gpi_image.offsets["g"][0]) + "_" + str(gpi_image.offsets["g"][1])
            b_offset = str(gpi_image.offsets["b"][0]) + "_" + str(gpi_image.offsets["b"][1])

            file_name = image_name + "_r" + r_offset + "_g" + g_offset + "_b" + b_offset + "_canny_edge_detection"
            save_image(processed_image, file_name, process_type)
        case "edge_images":
            gpi_image = GlassPlateImage(original_image, edge_detection_fn=gaussian_smoothening_edge_subtraction,
                                        to_edge=False)
            processed_image = gpi_image.aligned_edge_image(base)

            r_offset = str(gpi_image.offsets["r"][0]) + "_" + str(gpi_image.offsets["r"][1])
            g_offset = str(gpi_image.offsets["g"][0]) + "_" + str(gpi_image.offsets["g"][1])
            b_offset = str(gpi_image.offsets["b"][0]) + "_" + str(gpi_image.offsets["b"][1])

            file_name = image_name + "_r" + r_offset + "_g" + g_offset + "_b" + b_offset + "_gaussian_smoothening_edge_subtraction"
            save_image(processed_image, file_name, process_type)

            gpi_image = GlassPlateImage(original_image, edge_detection_fn=canny_edge_detection, to_edge=False)
            processed_image = gpi_image.aligned_edge_image(base)

            r_offset = str(gpi_image.offsets["r"][0]) + "_" + str(gpi_image.offsets["r"][1])
            g_offset = str(gpi_image.offsets["g"][0]) + "_" + str(gpi_image.offsets["g"][1])
            b_offset = str(gpi_image.offsets["b"][0]) + "_" + str(gpi_image.offsets["b"][1])

            file_name = image_name + "_r" + r_offset + "_g" + g_offset + "_b" + b_offset + "_canny_edge_detection"
            save_image(processed_image, file_name, process_type)
        case "edge_overlay_images":
            gpi_image = GlassPlateImage(original_image, edge_detection_fn=gaussian_smoothening_edge_subtraction,
                                        to_edge=False)
            processed_image = gpi_image.edges_on_base_image(base)

            r_offset = str(gpi_image.offsets["r"][0]) + "_" + str(gpi_image.offsets["r"][1])
            g_offset = str(gpi_image.offsets["g"][0]) + "_" + str(gpi_image.offsets["g"][1])
            b_offset = str(gpi_image.offsets["b"][0]) + "_" + str(gpi_image.offsets["b"][1])

            file_name = image_name + "_r" + r_offset + "_g" + g_offset + "_b" + b_offset + "_gaussian_smoothening_edge_subtraction"
            save_image(processed_image, file_name, process_type)

            gpi_image = GlassPlateImage(original_image, edge_detection_fn=canny_edge_detection, to_edge=False)
            processed_image = gpi_image.edges_on_base_image(base)

            r_offset = str(gpi_image.offsets["r"][0]) + "_" + str(gpi_image.offsets["r"][1])
            g_offset = str(gpi_image.offsets["g"][0]) + "_" + str(gpi_image.offsets["g"][1])
            b_offset = str(gpi_image.offsets["b"][0]) + "_" + str(gpi_image.offsets["b"][1])

            file_name = image_name + "_r" + r_offset + "_g" + g_offset + "_b" + b_offset + "_canny_edge_detection"
            save_image(processed_image, file_name, process_type)
        case "naive_colorized_images":
            gpi_image = GlassPlateImage(original_image, to_edge=False)
            processed_image = gpi_image.no_align_colorized()

            file_name = image_name
            save_image(processed_image, file_name, process_type)
        case "best_colorized_images":
            gpi_image = GlassPlateImage(original_image, to_edge=False)
            processed_image = gpi_image.best_aligned()

            r_offset = str(gpi_image.offsets["r"][0]) + "_" + str(gpi_image.offsets["r"][1])
            g_offset = str(gpi_image.offsets["g"][0]) + "_" + str(gpi_image.offsets["g"][1])
            b_offset = str(gpi_image.offsets["b"][0]) + "_" + str(gpi_image.offsets["b"][1])

            best_setting = gpi_image.best_alignment_settings[1]

            file_name = image_name + "_r" + r_offset + "_g" + g_offset + "_b" + b_offset + "_" + best_setting
            save_image(processed_image, file_name, process_type)
        case "gaussian_blurred_images":
            gpi_image = GlassPlateImage(original_image, to_edge=False, to_blurr=True)
            processed_image = gpi_image.best_aligned()

            r_offset = str(gpi_image.offsets["r"][0]) + "_" + str(gpi_image.offsets["r"][1])
            g_offset = str(gpi_image.offsets["g"][0]) + "_" + str(gpi_image.offsets["g"][1])
            b_offset = str(gpi_image.offsets["b"][0]) + "_" + str(gpi_image.offsets["b"][1])

            file_name = image_name + "_r" + r_offset + "_g" + g_offset + "_b" + b_offset + "_"
            save_image(processed_image, file_name, process_type)
        case _:
            raise TypeError


process_everything()
