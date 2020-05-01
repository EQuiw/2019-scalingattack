import numpy as np
import cvxpy as cp
import math

from attack.area_attack.AreaNormEnumType import AreaNormEnumType


def __get_weights_subblock(startrb, endrb, startcb, endcb, area):
    # real-valued coordinates
    # rx = r * scalex
    # rx2 = (r + 1) * scalex
    # cx = c * scaley
    # cx2 = (c + 1) * scaley
    rx = np.round(startrb, 5) # we do so to avoid that e.g. 255.000000001 gets 256 with math.ceil although just 255.
    rx2 = np.round(endrb, 5)
    cx = np.round(startcb, 5)
    cx2 = np.round(endcb, 5)

    # integer coordinates
    rx_left = math.floor(rx)
    rx_right = math.ceil(rx2)
    cx_left = math.floor(cx)
    cx_right = math.ceil(cx2)

    # if cx_right > src_img.shape[1]:
    #     cx_right = cx2 = src_img.shape[1]
    #     raise Exception("cx")
    # if rx_right > src_img.shape[0]:
    #     rx_right = rx2 = src_img.shape[0]
    #     raise Exception("rx")

    # now compute the weights for pixels that are taken into account completely and pixels that are not
    weights = np.ones((int(rx_right - rx_left), int(cx_right - cx_left)))
    if rx != rx_left:
        weights[0, :] = (math.ceil(rx) - rx)
    if rx2 != rx_right:
        weights[-1, :] = (rx2 - math.floor(rx2))
    if cx != cx_left:
        weights[:, 0] *= (math.ceil(cx) - cx)
    if cx2 != cx_right:
        weights[:, -1] *= (cx2 - math.floor(cx2))

    weights /= area
    assert np.round(np.sum(weights), 4) == 1.0

    return weights, rx_left, rx_right, cx_left, cx_right



def __area_direct_blockwise_nonintegerborders(tar_image: np.ndarray, src_img: np.ndarray,
                                            verbose: bool, eps: int, blockwise: float,
                                            attack_norm: AreaNormEnumType) -> np.ndarray:
    """
    The code here creates an attack image for the AREA scaling algorithm.
    To this end, it interprets the problem as an optimization problem,
    with min (x - src-img) s.th. 0 <= x <= 255 and sum(weight * x) = target-values where weight
     is the weight matrix for a respective sub-block...
    """

    scalex = src_img.shape[0] / tar_image.shape[0]
    scaley = src_img.shape[1] / tar_image.shape[1]

    # We focus on one direction for the following description:
    # Image is divided into sub-blocks. Each subblock has width: scalex, e.g. 2.5
    # To optimize, we should use a block that has an integer width. get it: This is a block

    def get_integer_blocksize(scalefactor, max_shape):
        for k in range(1, max_shape):
            if k * scalefactor % 2 == 0 or k * scalefactor % 1 == 0:
                return int(k * scalefactor)
        raise Exception("Should not happen")

    blockwidth_x = get_integer_blocksize(scalefactor=scalex, max_shape=src_img.shape[0])
    blockwidth_y = get_integer_blocksize(scalefactor=scaley, max_shape=src_img.shape[1])


    # Now, to reduce the optimization time, we want to stack multiple blocks together that
    # are optimized at the same time: our optimization blocks.

    # we need to divide the target image into blocks
    def get_opt_blocks(src_image_sh, block_size, min_blocks):
        blocks = src_image_sh / block_size
        return int(max(1, np.floor(blocks * min_blocks)))


    opt_block_step_widthx = get_opt_blocks(src_image_sh=src_img.shape[0], block_size=blockwidth_x, min_blocks=blockwise)
    opt_block_step_widthy = get_opt_blocks(src_image_sh=src_img.shape[1], block_size=blockwidth_y, min_blocks=blockwise)
    # opt_block_step_widthx specifies how many blocks are considered each time.

    area = scalex * scaley
    src_img_new = np.zeros(src_img.shape)

    subblocksx = int(opt_block_step_widthx * (blockwidth_x / scalex)) # total number of sub-blocks in each opt-block
    subblocksy = int(opt_block_step_widthy * (blockwidth_y / scaley))


    if verbose is True:
        print("We use {} blocks in one opti-block in x-dir (y-dir: {})".
              format(opt_block_step_widthx, opt_block_step_widthy))
        print("Each block consists of {} sub-blocks, each opti-block of {} sub-blocks, {} pixels each (y-dir: {},{},{})".
              format(blockwidth_x / scalex, subblocksx, scalex, blockwidth_y / scaley, subblocksy, scaley))


    for r in range(0, tar_image.shape[0], subblocksx):
        for c in range(0, tar_image.shape[1], subblocksy):

            rborder = np.minimum(tar_image.shape[0], r+subblocksx)
            cborder = np.minimum(tar_image.shape[1], c+subblocksy)

            target_value = tar_image[r:rborder, c:cborder]

            src_startx = int(r * scalex) # must be integer, as we iterate over blocks...
            src_endx = int(rborder * scalex)
            src_starty = int(c * scaley)
            src_endy = int(cborder * scaley)

            if verbose is True:
                print("x:x', y:y'; Tar: {}:{}, {}:{}; Src: {}:{}, {}:{}".
                      format(r,rborder,c,cborder,src_startx, src_endx, src_starty, src_endy))

            # objective function
            # define optimization problem
            novelpixels = cp.Variable((src_endx - src_startx, src_endy - src_starty))

            obj_vec = novelpixels - src_img[src_startx:src_endx, src_starty:src_endy]  # .reshape(-1)
            if attack_norm == AreaNormEnumType.L2:
                obj = (1 / 2) * cp.sum_squares(obj_vec)
            elif attack_norm == AreaNormEnumType.L1:
                obj = cp.sum(cp.abs(obj_vec))
            else: raise NotImplementedError()

            # constraint per sub-block
            constraints_subblocks = []
            for rtar, src_rb in enumerate(np.arange(0, src_endx - src_startx, scalex)):
                for ctar, src_cb in enumerate(np.arange(0, src_endy - src_starty, scaley)):
                    # rtar, ctar is index in current target image window
                    # src_rb, src_cb is index of subblock in current src image window

                    subblock_weights, rx_left, rx_right, cx_left, cx_right = __get_weights_subblock(
                                                            startrb=src_rb, endrb=src_rb+scalex,
                                                            startcb=src_cb, endcb=src_cb+scaley,
                                                            area=area)

                    t_w = cp.sum( cp.multiply(novelpixels[rx_left:rx_right, cx_left:cx_right], subblock_weights))
                    temp_constr = cp.abs(t_w - target_value[rtar, ctar]) <= eps

                    constraints_subblocks.append(temp_constr)


            constr2 = novelpixels <= 255
            constr3 = novelpixels >= 0
            prob = cp.Problem(cp.Minimize(obj), [*constraints_subblocks, constr2, constr3])

            # solve it
            try:
                prob.solve(solver=cp.OSQP)
            except:
                if verbose is True:
                    print("QSQP failed")
                try:
                    prob.solve(solver=cp.ECOS)
                except:
                    print("Could not solve with QSPS and ECOS at {}, {}".format(r, c))
                    raise Exception("Could not solve at {}, {}".format(r, c))
            if prob.status != cp.OPTIMAL and prob.status != cp.OPTIMAL_INACCURATE:
                print("Could only solve at {}, {} with status: {}".format(r, c, prob.status))
                raise Exception("Only solveable with infeasible/unbounded/optimal_inaccurate solution")


            src_img_new[src_startx:src_endx, src_starty:src_endy] = np.round(novelpixels.value)

    return src_img_new







def area_scale_attack_nonintegerborders(tar_image: np.ndarray,
                                        src_img: np.ndarray,
                                        verbose: bool,
                                        attack_norm: AreaNormEnumType,
                                        eps: int = 1,
                                        blockwise: float = 0.25) -> np.ndarray:
    """
    Creates an attack image, if downscaling, should look like target image...
    Works also for non-integer scaling factors, more general version of area_scale_integerborders.py.
    However, consider that this case is the more challenging part for the attacker, as she/he needs to
    compute a solution not only for one block that Area considers, but now for blocks that overlap.
    :param tar_image: target image
    :param src_img: source image
    :param verbose: if true, show progress
    :param attack_norm: If L1 or L2 should be used.
    :param eps: todo -- we only accept here one eps value != QuadrScaleAttack.
    :param blockwise: solve problem in larger blocks, is faster. Range (0, 1)
    :return: an attack image.
    """

    assert 0 < blockwise < 1
    assert isinstance(attack_norm, AreaNormEnumType)
    assert attack_norm == AreaNormEnumType.L1 or attack_norm == AreaNormEnumType.L2

    # separate scale for each channel
    if len(src_img.shape) == 2:
        # grayscale images with no channel dimension
        return __area_direct_blockwise_nonintegerborders(tar_image=tar_image, src_img=src_img,
                                                         verbose=verbose, eps=eps, blockwise=blockwise,
                                                         attack_norm=attack_norm)

    else:
        # color or grayscale images with channel dimension
        ret = np.zeros(src_img.shape).astype(np.uint8)
        for ch in range(src_img.shape[2]):
            ret[:, :, ch] = __area_direct_blockwise_nonintegerborders(tar_image=tar_image[:, :, ch], src_img=src_img[:, :, ch],
                                                                      verbose=verbose, eps=eps,
                                                                      blockwise = blockwise, attack_norm=attack_norm)
        return ret