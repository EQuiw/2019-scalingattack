import numpy as np
import typing
from scaling.ScalingApproach import ScalingApproach
import scaling.scale_utils
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
import cv2 as cv


class CVScaler(ScalingApproach):

    def __init__(self,
                 algorithm: typing.Union[int, SuppScalingAlgorithms],
                 src_image_shape: typing.Union[typing.Tuple[int, int], typing.Tuple[int, int, int]],
                 target_image_shape: typing.Union[typing.Tuple[int, int], typing.Tuple[int, int, int]]):

        super().__init__(algorithm, src_image_shape, target_image_shape)


    def scale_image_with(self, xin: np.ndarray, trows: int, tcols: int):
        return scaling.scale_utils.scale_cv2(xin=xin,
                                             trows=trows,
                                             tcols=tcols,
                                             alg=self.algorithm)


    def _convert_suppscalingalgorithm(self, algorithm: SuppScalingAlgorithms):
        if algorithm == SuppScalingAlgorithms.NEAREST:
            return cv.INTER_NEAREST
        elif algorithm == SuppScalingAlgorithms.LINEAR:
            return cv.INTER_LINEAR
        elif algorithm == SuppScalingAlgorithms.CUBIC:
            return cv.INTER_CUBIC
        elif algorithm == SuppScalingAlgorithms.LANCZOS:
            return cv.INTER_LANCZOS4
        elif algorithm == SuppScalingAlgorithms.AREA:
            return cv.INTER_AREA
        else:
            raise NotImplementedError()


