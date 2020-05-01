## get scale matrix ##
import numpy as np
import PIL
from PIL import Image
import cv2 as cv
import sys


def scale_pillow(xin, trows, tcols, alg=Image.LINEAR):

    dtype_before = xin.dtype

    if np.max(xin) < 1:
        print("Warning. Image might be not between 0 and 255", file=sys.stderr)

    # Ensure the pixel-values are between 0 and 255.
    img = np.clip(xin, 0, 255)

    # Convert the pixels to 8-bit bytes.
    # img = img.astype(np.uint8)

    # Create PIL-object from numpy array.
    img = PIL.Image.fromarray(img)

    # Resize the image.
    img_resized = img.resize((tcols, trows), alg)

    # Convert back
    # img_resized = np.float32(img_resized)
    img_resized = np.asarray(img_resized)

    # we want to have the same output type as input type; astype rounds towards zero, so we use round before.
    if np.issubdtype(dtype_before, np.integer):
        img_resized = np.round(img_resized).astype(dtype_before)

    # print(img_resized.dtype)
    return img_resized


def scale_cv2(xin, trows, tcols, alg=cv.INTER_CUBIC):
    res = cv.resize(xin, (tcols, trows), interpolation=alg)
    return res
