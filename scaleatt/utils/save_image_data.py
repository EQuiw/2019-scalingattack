from PIL import Image
import numpy as np
import scipy.misc


def save_jpeg_image(out_img: np.ndarray, image_path: str, quality: int = 75) -> None:
    assert out_img.dtype == np.uint8
    im = Image.fromarray(out_img)
    im.save(image_path, quality=quality)


def save_png_image(out_img: np.ndarray, image_path: str) -> None:
    assert out_img.dtype == np.uint8
    scipy.misc.toimage(out_img, cmin=0, cmax=255).save(image_path)
