import numpy as np
import typing

from scaling.ScalingApproach import ScalingApproach
from defenses.prevention.PreventionDefense import PreventionDefense
from defenses.detection.fourier.FourierPeakMatrixCollector import FourierPeakMatrixCollector


from defenses.prevention.cythmodule.medianfiltering import median_filtering_cython


class MedianFilteringDefense(PreventionDefense):

    def __init__(self, verbose: bool,
                 scaler_approach: ScalingApproach,
                 fourierpeakmatrixcollector: FourierPeakMatrixCollector,
                 bandwidth: typing.Optional[int],
                 usecython: bool):
        """
        :param verbose:
        :param scaler_approach: scaler approach
        :param fourierpeakmatrixcollector: collector that saves an ideal attack image, used to determine
        which pixels are considered by scaling algorithm in spatial space.
        :param bandwidth: if none, we set bandwidth such that with k = scaling ratio vert., k2 = ratio horiz,
        we consider +/- k pixels vertically and +/- k2 pixels horizontally around each considered pixels.
        If bandwidth is an integer, we divide k by bandwidth and can adjust the bandwidth accordingly.
        :param usecython: bool, use faster cython version.
        """
        super().__init__(verbose, scaler_approach)

        # if not isinstance(self.scaler_approach, PythonReImplementation) or not isinstance(self.scaler_approach, ScalingApproach):
        #     raise NotImplementedError("The prevention defense based on median filter requires " +
        #         "that we know where pixels are used by scaling algorithm. Thus, we need a ScalerApproach + PythonReImplementation")

        bandwidthfactor = 1 if bandwidth is None else bandwidth

        src_shape0 = self.scaler_approach.cl_matrix.shape[1]
        tar_shape0 = self.scaler_approach.cl_matrix.shape[0]
        src_shape1 = self.scaler_approach.cr_matrix.shape[0]
        tar_shape1 = self.scaler_approach.cr_matrix.shape[1]

        scale_factor_hz = src_shape1 / tar_shape1
        scale_factor_vt = src_shape0 / tar_shape0

        self.bandwidth: typing.Tuple[int, int] = (int(np.floor(scale_factor_vt/bandwidthfactor)),
                                                  int(np.floor(scale_factor_hz/bandwidthfactor)))

        if self.verbose is True:
            print("Kernel size of filter in one direction:", self.bandwidth)

        self.fourierpeakmatrixcollector: FourierPeakMatrixCollector = fourierpeakmatrixcollector
        self.usecython = usecython


    def make_image_secure(self, att_image: np.ndarray) -> np.ndarray:

        # I. Get considered pixels via fourierpeakcollector
        dir_attack_image = self.fourierpeakmatrixcollector.get(scaler_approach=self.scaler_approach)
        binary_mask_indices = np.where(dir_attack_image != 255)
        binary_mask = np.zeros((self.scaler_approach.cl_matrix.shape[1], self.scaler_approach.cr_matrix.shape[0]))
        binary_mask[binary_mask_indices] = 1


        # II. go over each channel if necessary
        if len(att_image.shape) == 2:
            if self.usecython is False:
                r= self.__apply_median_filtering(att_image=att_image, binary_mask = binary_mask)
            else:
                r = self.__apply_median_filtering_cython(att_image=att_image, binary_mask=binary_mask)
            return r.astype(np.uint8)

        else:
            filtered_att_image = np.zeros(att_image.shape)
            for ch in range(att_image.shape[2]):
                if self.verbose is True:
                    print("Channel:", ch)

                if self.usecython is False:
                    re = self.__apply_median_filtering(att_image=att_image[:,:,ch], binary_mask = binary_mask)
                else:
                    re = self.__apply_median_filtering_cython(att_image=att_image[:, :, ch], binary_mask=binary_mask)

                filtered_att_image[:, :, ch] = re
            return filtered_att_image.astype(np.uint8)


    def __apply_median_filtering(self, att_image: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:

        filtered_attack_image = np.copy(att_image)
        positions = np.where(binary_mask==1)

        # we convert to float for inserting nans, then we insert nan at all locations that are marked in binary-mask.
        #   later, when we compute the median around each marked location, we can very simply ignore all other
        #   marked locations that are inside the window
        base_attack_image = np.copy(att_image)
        base_attack_image = base_attack_image.astype('float64')
        assert np.any(np.isnan(base_attack_image)) == False
        base_attack_image[positions] = np.nan

        # apply median filter
        xpos = positions[0]
        ypos = positions[1]
        for pix_src_r, pix_src_c in zip(xpos, ypos):

            ix_l = max(0, pix_src_r - self.bandwidth[0])
            ix_r = min(pix_src_r + self.bandwidth[0] + 1, filtered_attack_image.shape[0] )
            jx_u = max(0, pix_src_c - self.bandwidth[1])
            jx_b = min(pix_src_c + self.bandwidth[1] + 1, filtered_attack_image.shape[1] )

            # filtered_attack_image[pix_src_r, pix_src_c] = np.nanmedian(base_attack_image[ix_l:ix_r, jx_u:jx_b])
            filtered_attack_image[pix_src_r, pix_src_c] = MedianFilteringDefense.get_median_nan(base_attack_image[ix_l:ix_r, jx_u:jx_b])

        return filtered_attack_image


    def __apply_median_filtering_cython(self, att_image: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
        """
        This is a cython wrapper that calls the respective cython function. Much faster than the Python version.
        :param att_image: image under investigation
        :param binary_mask: binary mask: pixels that are considered
        :return: filtered image
        """

        filtered_attack_image = np.copy(att_image)
        positions = np.where(binary_mask == 1)
        xpos = positions[0]
        ypos = positions[1]

        res = median_filtering_cython(att_image, filtered_attack_image,
                                      binary_mask.astype(np.uint8), xpos, ypos,
                                      self.bandwidth[0], self.bandwidth[1])
        return np.array(res) # cython returns memoryview..


    @staticmethod
    def get_median_nan(arr: np.ndarray):
        """
        Replaces np.nanmedian which is slower, and we do not compute the median as weighted average
        if array has an even length, but take the middle as for uneven length.
        So for even lenght, usually we would compute s.th. like e.g. (e1+e2)/2) ...
        Now we simply take the e1 value.
        :param arr: array where nan values will be ignored.
        :return: median
        """
        arr = arr[~np.isnan(arr)]
        arr.flatten()
        arr.sort()
        if len(arr) % 2 == 0:
            median_it1 = int(len(arr) / 2 - 1)
        else:
            median_it1 = int(len(arr) / 2)
        return arr[median_it1]