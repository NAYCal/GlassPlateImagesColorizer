from enum import Enum
import os
import skimage as sk
import skimage.io as skio


class DefaultImages(Enum):
    CATHEDRAL = "cathedral.jpg"
    CHURCH = "church.tif"
    EMIR = "emir.tif"
    HARVESTERS = "harvesters.tif"
    ICON = "icon.tif"
    LADY = "lady.tif"
    MELONS = "melons.tif"
    MONASTERY = "monastery.tif"
    ONION_CHURCH = "onion_church.tif"
    SCULPTURE = "sculpture.tif"
    SELF_PORTRAIT = "self_portrait.tif"
    THREE_GENERATIONS = "three_generations.tif"
    TOBOLSK = "tobolsk.jpg"
    TRAIN = "train.tif"

    def get_image(self):
        image_filename = self.value
        script_dir = os.path.dirname(os.path.abspath(__file__))
        module_dir = os.path.dirname(script_dir)
        project_dir = os.path.dirname(module_dir) + "/data/"

        # Create the full path to the image
        image_path = os.path.join(project_dir, image_filename)
        image = skio.imread(image_path)
        image = sk.img_as_float(image)

        return image


if __name__ == "__main__":
    skio.imshow(DefaultImages.CATHEDRAL.get_image())
    skio.show()
