"""
The code here creates an attack image for the AREA scaling algorithm.
To this end, it interprets the problem as an optimization problem,
with min (x - src-img) s.th. 0 <= x <= 255 and 1/n * sum(x) = target-values where n is the respective block
in source image whose average value is used to obtain the target value during the downscaling area operation.
"""

import numpy as np
import cvxpy as cp

from attack.area_attack.AreaNormEnumType import AreaNormEnumType


def area_direct_blockwise_int(tar_image: np.ndarray, src_img: np.ndarray, ksizex:int, ksizey: int, verbose: bool,
                              eps: int, attack_norm: AreaNormEnumType):
    """
    L1/L2 attack variant against Area scaling.
    Divides the image into several larger blocks where optimization problem is solved to get attack image.
    Is faster than 'area_direct' (which solves each block)...
    """
    src_img_new = np.zeros(src_img.shape)
    use_l2: bool = (attack_norm == AreaNormEnumType.L2) # otherwise, use L1

    # we need to divide the target image into blocks
    def get_step(tar_image_sh):
        for divi in range(20, 2, -1):
            if tar_image_sh % divi == 0:
                return int(tar_image_sh / divi)

    stepx = get_step(tar_image_sh=tar_image.shape[0])
    stepy = get_step(tar_image_sh=tar_image.shape[1])

    for r in range(0, tar_image.shape[0], stepx):
        for c in range(0, tar_image.shape[1], stepy):
            if verbose is True and c == 0:
                print("Iteration: {}, {}".format(r, c))

            target_value = tar_image[r:(r + stepx), c:(c + stepy)]

            # define optimization problem
            novelpixels = cp.Variable((ksizex * stepx, ksizey * stepy))

            # get region in source image
            startx = r * ksizex
            endx = ((r + stepx) * ksizex)
            starty = c * ksizey
            endy = ((c + stepy) * ksizey)

            # objective function
            obj_vec = novelpixels - src_img[startx:endx, starty:endy]  # .reshape(-1)
            if use_l2:
                obj = (1 / 2) * cp.sum_squares(obj_vec)
            else:
                obj = cp.sum(cp.abs(obj_vec))

            # constraint 1:
            constrs = []
            for rb, rtarind in zip(range(0, endx - startx, ksizex), range(stepx)):
                for cb, ctarind in zip(range(0, endy - starty, ksizey), range(stepy)):
                    # we use zip to obtain the index in src-img (rb, cd) first and
                    #   index in target-image (rtarind, ctarind) as 2nd object.
                    # print(rb, cb, rtarind, ctarind)

                    temp_constr = cp.abs(cp.sum(novelpixels[rb:(rb + ksizex), cb:(cb + ksizey)]) - \
                                         (target_value[rtarind, ctarind] * ksizex * ksizey)) <= eps
                    constrs.append(temp_constr)

            constr2 = novelpixels <= 255
            constr3 = novelpixels >= 0

            prob = cp.Problem(cp.Minimize(obj), [*constrs, constr2, constr3])

            # solve it, we first try default solver, and then if not possible (rarely the case), we try another.
            # check the docu: https://www.cvxpy.org/tutorial/intro/index.html#infeasible-and-unbounded-problems
            try:
                prob.solve()
            except:
                if verbose is True:
                    print("QSQP failed at {}, {}".format(r, c))
                try:
                    prob.solve(solver=cp.ECOS)
                except:
                    print("Could not solve with QSPS and ECOS at {}, {}".format(r, c))
                    raise Exception("Could not solve at {}, {}".format(r, c))

            if prob.status != cp.OPTIMAL and prob.status != cp.OPTIMAL_INACCURATE:
                print("Could only solve at {}, {} with status: {}".format(r, c, prob.status))
                raise Exception("Only solveable with infeasible/unbounded/optimal_inaccurate solution")

            assert prob is not None and novelpixels.value is not None # actually not needed, just to ensure..
            # print(np.round(novelpixels.value.reshape((ksizex, ksizey))))
            src_img_new[startx:endx, starty:endy] = np.round(novelpixels.value)


    return src_img_new


def area_direct_int(tar_image: np.ndarray, src_img: np.ndarray, ksizex:int, ksizey: int, verbose: bool, eps: int,
                    attack_norm: AreaNormEnumType):

    src_img_new = np.zeros(src_img.shape)
    use_l2: bool = (attack_norm == AreaNormEnumType.L2) # otherwise, use L1

    for r in range(tar_image.shape[0]):
        for c in range(tar_image.shape[1]):
            if verbose is True and r % 20 == 0 and c == 0:
                print("Iteration: {}, {}".format(r,c))

            target_value = tar_image[r,c]

            # define optimization problem
            novelpixels = cp.Variable(ksizex*ksizey)
            # ident = np.identity(ksizex*ksizey)

            startx = r*ksizex
            endx = ((r+1)*ksizex)
            starty = c*ksizey
            endy = ((c+1)*ksizey)
            obj_vec = novelpixels - src_img[ startx:endx, starty:endy ].reshape(-1)

            if use_l2:
                obj = (1 / 2) * cp.quad_form(obj_vec, np.identity(ksizex*ksizey))
            else:
                obj = cp.sum(cp.abs(obj_vec))

            # constr1 = cp.sum(novelpixels) == (target_value * ksizex*ksizey)
            constr1 = cp.abs(cp.sum(novelpixels) - (target_value * ksizex*ksizey)) <= eps
            constr2 = novelpixels <= 255
            constr3 = novelpixels >= 0

            prob = cp.Problem(cp.Minimize(obj), [constr1, constr2, constr3])

            try:
                prob.solve()
            except:
                if verbose is True:
                    print("QSQP failed at {}, {}".format(r, c))
                try:
                    prob.solve(solver=cp.ECOS)
                except:
                    print("Could not solve with QSPS and ECOS at {}, {}".format(r, c))
                    raise Exception("Could not solve at {}, {}".format(r, c))

            if prob.status != cp.OPTIMAL and prob.status != cp.OPTIMAL_INACCURATE:
                print("Could only solve at {}, {} with status: {}".format(r, c, prob.status))
                raise Exception("Only solveable with infeasible/unbounded/optimal_inaccurate solution")
            assert prob is not None and novelpixels.value is not None  # actually not needed, just to ensure..

            # print(np.round(novelpixels.value.reshape((ksizex, ksizey))))
            src_img_new[ startx:endx, starty:endy ] = np.round(novelpixels.value.reshape((ksizex, ksizey)))

    return src_img_new


def area_scale_attack_integerborders(tar_image: np.ndarray, src_img: np.ndarray, verbose: bool,
                                     attack_norm: AreaNormEnumType, eps: int = 1, blockwise: bool = True):
    """
    Creates an attack image, if downscaling, should look like target image...
    Works only for integer downscaling factors
    :param tar_image: target image
    :param src_img: source image
    :param verbose: if true, show progress.
    :param attack_norm: chosen L1, L2 norm..
    :param eps: epsilon for optimization problem. todo -- we only accept here one eps value != QuadrScaleAttack.
    :param blockwise: solve problem in larger blocks, is faster.
    :return: attack image
    """

    scalex = int(src_img.shape[0] / tar_image.shape[0])
    scaley = int(src_img.shape[1] / tar_image.shape[1])

    if scalex != round(src_img.shape[0] / tar_image.shape[0], 2) or\
        scaley != round(src_img.shape[1] / tar_image.shape[1], 2):
        raise NotImplementedError("This method expects that scaling ratio is an integer")

    assert isinstance(attack_norm, AreaNormEnumType)
    assert attack_norm == AreaNormEnumType.L1 or attack_norm == AreaNormEnumType.L2

    # separate scale for each channel
    if len(src_img.shape) == 2:
        # grayscale images with no channel dimension
        if blockwise is True:
            return area_direct_blockwise_int(tar_image=tar_image, src_img=src_img, ksizex=scalex, ksizey=scaley,
                                             verbose=verbose, eps=eps, attack_norm=attack_norm)
        else:
            return area_direct_int(tar_image=tar_image, src_img=src_img, ksizex=scalex, ksizey=scaley,
                                   verbose=verbose, eps=eps, attack_norm=attack_norm)
    else:
        # color or grayscale images with channel dimension
        ret = np.zeros(src_img.shape).astype(np.uint8)
        for ch in range(src_img.shape[2]):
            if blockwise is True:
                ret[:, :, ch] = area_direct_blockwise_int(tar_image=tar_image[:, :, ch], src_img=src_img[:, :, ch],
                                                          ksizex=scalex, ksizey=scaley, verbose=verbose, eps=eps,
                                                          attack_norm=attack_norm)
            else:
                ret[:, :, ch] = area_direct_int(tar_image=tar_image[:, :, ch], src_img=src_img[:, :, ch], ksizex=scalex,
                                                ksizey=scaley, verbose=verbose, eps=eps, attack_norm=attack_norm)
        return ret


