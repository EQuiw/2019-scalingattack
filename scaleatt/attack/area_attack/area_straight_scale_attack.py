import numpy as np

def _area_straight_direct(tar_image: np.ndarray, src_img: np.ndarray, ksizex:int, ksizey: int, verbose: bool, permutation: bool):
    """
    L0-based attack against Area Scaling. See our paper for the description of the algorithm.
    :param tar_image: target image
    :param src_img: source image
    :param ksizex: area block size, assumes an integer scaling ratio.
    :param ksizey: area block size, assumes an integer scaling ratio.
    :param verbose: show progress if true
    :param permutation: if true, use permutation.
    :return: attack image, rewritten pixels
    """
    src_img_new = src_img.copy()
    rewritten_pixels = np.zeros(tar_image.shape)

    for r in range(tar_image.shape[0]):
        for c in range(tar_image.shape[1]):
            if verbose is True and r % 20 == 0 and c == 0:
                print("Iteration: {}, {}".format(r,c))

            startx = int( r * ksizex )
            endx = int( ((r + 1) * ksizex) )
            starty = int( c * ksizey )
            endy = int( ((c + 1) * ksizey) )

            region = src_img_new[startx:endx, starty:endy]
            region_mean = region.mean()

            if tar_image[r, c] == region_mean:
                continue
            elif tar_image[r, c] > region_mean:
                add = True
                mode_add = 255
            else:
                add = False
                mode_add = 0

            finished = False

            list_of_pairs = [(x,y) for x in range(region.shape[0]) for y in range(region.shape[1])]
            if permutation is True:
                np.random.seed(32 + 19*r + c)
                np.random.shuffle(list_of_pairs)

            nochanged = 0
            for ri, ci in list_of_pairs:
                if finished is True:
                    break

                next_value = region_mean - \
                             src_img_new[ri + startx, ci + starty] / region.size + \
                             mode_add / region.size


                if add is True and next_value > tar_image[r, c]:
                    mode_add = tar_image[r, c] * region.size - \
                               region_mean * region.size + \
                               src_img_new[ri + startx, ci + starty]
                    finished = True
                if add is False and next_value < tar_image[r, c]:
                    mode_add = tar_image[r, c] * region.size - \
                               region_mean * region.size + \
                               src_img_new[ri + startx, ci + starty]
                    finished = True


                region_mean = region_mean - \
                              src_img_new[ri + startx, ci + starty] / region.size + \
                              mode_add / region.size
                src_img_new[ri + startx, ci + starty] = mode_add
                nochanged += 1


            rewritten_pixels[r, c] = nochanged / len(list_of_pairs)

    return src_img_new, rewritten_pixels





def area_straight_scale_attack(tar_image: np.ndarray, src_img: np.ndarray, verbose: bool, permutation: bool):
    """
    Creates an attack image, if downscaling, should look like target image...
    :param tar_image: target image
    :param src_img: source image
    :param verbose: show progress if true
    :param permutation: if true, list of possible pixels that are rewritten does not start at the top-left corner,
    but is shuffled. We often achieve better results visually if we set permutation=False.
    :return:
    a) attack image,
    b) rewritten pixels (For each pixel in the target image, the percentage of changed pixels in each block of the
    source image to move the average towards the respective target value).
    """

    scalex = int(src_img.shape[0] / tar_image.shape[0])
    scaley = int(src_img.shape[1] / tar_image.shape[1])

    if scalex != round(src_img.shape[0] / tar_image.shape[0], 2) or\
        scaley != round(src_img.shape[1] / tar_image.shape[1], 2):
        raise NotImplementedError("This method expects that scaling ratio is an integer")

    # separate scale for each channel
    if len(src_img.shape) == 2:
        # grayscale images with no channel dimension
        return _area_straight_direct(tar_image=tar_image, src_img=src_img, ksizex=scalex, ksizey=scaley, verbose=verbose, permutation=permutation)
    else:
        # color or grayscale images with channel dimension
        ret = np.zeros(src_img.shape).astype(np.uint8)
        ret2 = np.zeros(tar_image.shape)
        for ch in range(src_img.shape[2]):
            ret[:,:,ch], ret2[:,:,ch] = _area_straight_direct(tar_image=tar_image[:,:,ch], src_img=src_img[:,:,ch],
                                                ksizex=scalex, ksizey=scaley, verbose=verbose, permutation=permutation)
        return ret, ret2
