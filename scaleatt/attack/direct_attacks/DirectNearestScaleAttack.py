from attack.ScaleAttackStrategy import ScaleAttackStrategy
import numpy as np
import typing
import sys
import cv2 as cv

from scaling.ScalingApproach import ScalingApproach


class DirectNearestScaleAttack(ScaleAttackStrategy):
    """
    A direct version of an image-scaling attack without solving an optimization problem.
    We can derive for any scaling algorithm + library such a direct version due to our root-cause analysis.
    In this way, we do not need to cast the problem as optimization problem.
    TODO - not tested completely, sometimes some columns are not correctly set, rounding errors?
    """

    def __init__(self, verbose: bool):
        super().__init__(verbose)
        self.round_to_integer = True


    # @Overwrite
    def _attack_ononedimension(self, src_image: np.ndarray, target_image: np.ndarray,
               scaler_approach: ScalingApproach) \
            -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:


        # assert scaler_approach.algorithm == cv.INTER_NEAREST
        if scaler_approach.algorithm != cv.INTER_NEAREST:
            print("Warning. No inter_nearest used as interpolation", file=sys.stderr)

        dir_attack_image = np.copy(src_image)

        scale_factor_hz = src_image.shape[1] / target_image.shape[1]
        scale_factor_vt = src_image.shape[0] / target_image.shape[0]

        for r in range(0, target_image.shape[0]):
            for c in range(0, target_image.shape[1]):
                dir_attack_image[int(np.floor(r * scale_factor_vt)), int(np.floor(c * scale_factor_hz))] = \
                    target_image[r, c]

        return dir_attack_image, np.zeros(1), np.zeros(1)
