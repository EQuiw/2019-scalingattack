# get image examples

import cv2 as cv

from utils.plot_image_utils import plot_images2
import scaling.scale_utils as scale_utils
import skimage.data


########################## Load image ##########################

def load_image_examples(img_src: int, plot_loaded: bool = True):
    """
    Load simple examples to demonstrate attack.
    :param img_src: number to select pair.
    :param plot_loaded: if true, images are plotted.
    :return: source image, target image.
    """

    if img_src == 1:
        src_image_example = skimage.data.astronaut()
        src_image_example = scale_utils.scale_cv2(src_image_example, 1024, 1024)

        tar_image_example = skimage.data.rocket()
        tar_image_example = scale_utils.scale_cv2(tar_image_example, 96, 96)

    elif img_src == 2:
        src_image_example = skimage.data.coffee()

        tar_image_example = skimage.data.chelsea()
        tar_image_example = scale_utils.scale_cv2(tar_image_example, 60, 90)

    # elif img_src == 3:
    # here, it would be easy to add a new image example given by a library.

    else:
        raise NotImplementedError()

    if plot_loaded is True:
        plot_images2(src_image_example, tar_image_example)

    return src_image_example, tar_image_example


def load_color_image_example_from_disk(img_src_path: str, img_tar_path: str):
    """
    Loads src, and target RGB images by providing the path to both images.
    """
    # do not forget to swap axis for images loaded by CV, as it saves images in BGR format (not RGB).
    src_image_example = cv.imread(img_src_path)
    src_image_example = cv.cvtColor(src_image_example, cv.COLOR_BGR2RGB)
    tar_image_example = cv.imread(img_tar_path)
    tar_image_example = cv.cvtColor(tar_image_example, cv.COLOR_BGR2RGB)

    return src_image_example, tar_image_example