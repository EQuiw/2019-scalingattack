import numpy as np
import typing
from PIL import Image

from scaling.ScalingApproach import ScalingApproach
import scaling.scale_utils
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms


class PillowScaler(ScalingApproach):

    def __init__(self,
                 algorithm: typing.Union[int, SuppScalingAlgorithms],
                 src_image_shape: typing.Union[typing.Tuple[int, int], typing.Tuple[int, int, int]],
                 target_image_shape: typing.Union[typing.Tuple[int, int], typing.Tuple[int, int, int]]):

        super().__init__(algorithm, src_image_shape, target_image_shape)


    def scale_image_with(self, xin: np.ndarray, trows: int, tcols: int):
        return scaling.scale_utils.scale_pillow(xin=xin,
                                             trows=trows,
                                             tcols=tcols,
                                             alg=self.algorithm)


    def _convert_suppscalingalgorithm(self, algorithm: SuppScalingAlgorithms):
        if algorithm == SuppScalingAlgorithms.NEAREST:
            return Image.NEAREST
        elif algorithm == SuppScalingAlgorithms.LINEAR:
            return Image.BILINEAR
        elif algorithm == SuppScalingAlgorithms.CUBIC:
            return Image.CUBIC
        elif algorithm == SuppScalingAlgorithms.LANCZOS:
            return Image.LANCZOS
        elif algorithm == SuppScalingAlgorithms.AREA:
            return Image.BOX
        else:
            raise NotImplementedError()