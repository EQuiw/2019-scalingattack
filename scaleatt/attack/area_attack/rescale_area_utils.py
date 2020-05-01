import numpy as np
import scaling.scale_utils as scale_utils


def rescale_to_integer(noninteger: bool, src_image: np.ndarray, tar_image_shape: tuple, row: int) -> np.ndarray:
    """
    Scales src_image to multiple of tar_image_shape, so that scaling ratio becomes an integer.
    :param noninteger: if noninteger is true, src-image is not scaled, if false, it is scaled.
    :param src_image: source image
    :param tar_image_shape: tar-image shape
    :param row: current row from dataset, otherwise pass 0 or any other int, only used for message in case of error.
    :return: (re-scaled) src-image.
    """
    if noninteger is True:
        src_image_example = src_image.copy()
    else:
        # Scale source image so that scaling factor becomes integer
        scalex = np.floor(src_image.shape[0] / tar_image_shape[0])
        scaley = np.floor(src_image.shape[1] / tar_image_shape[1])
        if scalex < 2.0 or scaley < 2.0:
            print("scale factor skipping", row)
            raise Exception("scale factor skipping")

        src_image_example = scale_utils.scale_cv2(src_image,
                                                  int(scalex * tar_image_shape[0]),
                                                  int(scaley * tar_image_shape[1]))

    return src_image_example