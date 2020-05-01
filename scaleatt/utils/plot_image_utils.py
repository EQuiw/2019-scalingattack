import numpy as np
import matplotlib.pyplot as plt
import typing


def plot_images1(img1):
    # assert np.min(img1) >= -.49 and np.max(img1) <= .5 + 1e-9
    plt.subplot(1, 1, 1)
    plt.imshow(img1 + 0, cmap=plt.cm.gray)  # , vmin=0, vmax=1)
    plt.title("Image", fontsize=7)
    plt.show()


def plot_images2(img1, img2=None, title=""):
    # assert np.min(img1) >= -0.1 and np.max(img1) <= 1.1
    plt.subplot(1, 2, 1)
    plt.imshow(img1, vmin=0, vmax=255, cmap=plt.cm.gray)
    plt.title(" Image/" + title, fontsize=7)
    if img2 is not None:
        plt.subplot(1, 2, 2)
        plt.imshow(img2, vmin=0, vmax=255, cmap=plt.cm.gray)
        plt.title("Image2", fontsize=7)
    plt.show()


def plot_images(imgs: list, titles: typing.List[str], rows: int):

    cols: int = int(np.ceil(len(imgs)/rows))

    for i, cur_img in enumerate(imgs):
        plt.subplot(rows, cols, i+1)
        plt.imshow(imgs[i], vmin=0, vmax=255,  cmap=plt.cm.gray)
        plt.title(titles[i], fontsize=8)

    plt.show()


def plot_images1_actual_size(img: np.ndarray):

    dpi = 100
    height, width = img.shape[0], img.shape[1]

    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.imshow(img, cmap='gray')
    plt.show()



def plot_images_in_actual_size(imgs: typing.List[np.ndarray], titles: typing.List[str], rows: int) -> None:
    """
    Assumes that all images in list have same size. Otherwise, images are scaled.
    :param imgs: list of images
    :param titles: list of titles for respective images
    :param rows: the number of rows for the sub figs
    """

    margin = 50  # pixels
    spacing = 35  # pixels
    dpi = 100.  # dots per inch

    cols: int = int(np.ceil(len(imgs) / rows))

    width = (imgs[0].shape[1] * cols + 2 * margin + spacing) / dpi  # inches
    height = (imgs[0].shape[0] * rows + 2 * margin + spacing) / dpi

    left = margin / dpi / width  # axes ratio
    bottom = margin / dpi / height
    wspace = spacing / float(200)

    fig, axes = plt.subplots(rows, cols, figsize=(width, height), dpi=dpi)
    fig.subplots_adjust(left=left, bottom=bottom, right=1. - left, top=1. - bottom,
                        wspace=wspace, hspace=wspace)

    for ax, im, name in zip(axes.flatten(), imgs, titles):
        ax.axis('off')
        ax.set_title('{}'.format(name))
        ax.imshow(im, cmap='gray')

    plt.show()