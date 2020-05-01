from abc import ABC, abstractmethod
import numpy as np
import typing

from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms


class ScalingApproach(ABC):
    """
    The setup for a scaling approach.
    Includes the library (e.g. PIL or OpenCV) and the interpolation algorithm.
    This class saves the CL, CR, interpolation algorithm and the image sizes before and after.
    The reason is that CL and CR are only valid for a fixed size in combination with the right interpolation algorithm,
    so that we should not save them separately.
    """

    def __init__(self,
                 algorithm: typing.Union[int, SuppScalingAlgorithms],
                 src_image_shape: typing.Union[typing.Tuple[int, int], typing.Tuple[int, int, int]],
                 target_image_shape: typing.Union[typing.Tuple[int, int], typing.Tuple[int, int, int]]):
        """
        :param algorithm: can be 'int' if this value corresponds to the algorithm value from the library,
        e.g. we use OpenCV and pass INTER_LINEAR (which is int); or this is SuppScalingAlgorithms, so that
        we determine the necessary library parameter automatically.
        :param src_image_shape:
        :param target_image_shape:
        """

        assert isinstance(algorithm, int) or isinstance(algorithm, SuppScalingAlgorithms)
        if isinstance(algorithm, SuppScalingAlgorithms):
            algorithm = self._convert_suppscalingalgorithm(algorithm)
        self.algorithm = algorithm

        self.src_image_shape = src_image_shape
        self.target_image_shape = target_image_shape

        self.cl_matrix, self.cr_matrix = self.__get_matrix_cr_cl()



    @abstractmethod
    def scale_image_with(self, xin: np.ndarray, trows: int, tcols: int) -> np.ndarray:
        pass


    def scale_image(self, xin: np.ndarray) -> np.ndarray:
        """
        Scales image w.r.t to target image shape
        """
        return self.scale_image_with(xin=xin, trows=self.target_image_shape[0], tcols=self.target_image_shape[1])


    @abstractmethod
    def _convert_suppscalingalgorithm(self, algorithm: SuppScalingAlgorithms):
        """
        Convert enum for common algorithm specification to scaling library value.
        :param algorithm:
        :return:
        """
        pass


    def __get_scale_cr_cl(self, sh, trows, tcols):
        """
        Helper Function for Coefficients Recovery (to get CL and CR matrix)
        :param sh: shape
        :return:
        """
        mint = 255
        im_max = mint * np.identity(sh)

        im_max_scaled = self.scale_image_with(xin=im_max, trows=trows, tcols=tcols)

        Cm = im_max_scaled / 255
        return Cm


    def __get_matrix_cr_cl(self):
        """
        Coefficients Recovery
        :return: CL and CR matrix
        """

        CL = self.__get_scale_cr_cl(sh=self.src_image_shape[0],
                                    trows=self.target_image_shape[0],
                                    tcols=self.src_image_shape[0])
        CR = self.__get_scale_cr_cl(sh=self.src_image_shape[1],
                                    tcols=self.target_image_shape[1],
                                    trows=self.src_image_shape[1])
        # print(CL.shape, CR.shape)

        # normalize
        CL = CL / CL.sum(axis=1)[:, np.newaxis]
        CR = CR / CR.sum(axis=0)[np.newaxis, :]

        return CL, CR


    def get_unique_approach_identifier(self) -> str:
        return str(self.algorithm) + "-" + str(type(self).__name__)