from abc import ABC, abstractmethod
import numpy as np
import typing
# import sys

from scaling.ScalingApproach import ScalingApproach


class ScaleAttackStrategy(ABC):
    """
    Interface for image-scaling attack.
    """

    def __init__(self, verbose: bool):
        self.verbose = verbose


    def attack(self, src_image: np.ndarray, target_image: np.ndarray,
               scaler_approach: ScalingApproach) \
            -> typing.Tuple[np.ndarray, typing.List[np.ndarray], typing.List[np.ndarray]]:
        """
        Peform the scaling attack. Input can have the shape [rows, cols, channels] for color images
        or [rows, cols] for gray-scale images.
        :param src_image: source image of attack
        :param target_image: target image of attack
        :param scaler_approach: scaling approach object, defines library and algorithm to be used..
        :return: resulting attack image (as np.uint8), list of objective function values, list of objective function values
        """

        # ensure that range is in [0, 255]
        assert np.max(src_image) < 255.01 and np.min(src_image) >= -0.0001
        assert np.max(target_image) < 255.01 and np.min(src_image) >= -0.0001
        if src_image.dtype != np.uint8 or target_image.dtype != np.uint8:
            raise Exception("Source or target image have dtype != np.uint8; actually the attack also works "
                            "with float values, as long as they are in the range [0, 255]. So if you know "
                            "what you are doing, you can remove this exception. But keep in mind that the "
                            "return type of the attack image will be np.uint8 even if the input is e.g. np.float64. "
                            "If you need other ranges, you need to adapt our implementation, but the attack works "
                            "for any range in theory.")
            # print("Source or target image have dtype != np.uint8", file=sys.stderr)


        if len(src_image.shape) == 2:
            if self.verbose is True:
                print("*** 2D ***")
            result_attack_image, opt_values1, opt_values2 = self._attack_ononedimension(src_image=src_image,
                                              target_image=target_image,
                                              scaler_approach=scaler_approach)
            result_attack_image = result_attack_image.astype(np.uint8)
            return result_attack_image, [opt_values1], [opt_values2] # ensure that output format is correct

        else:
            result_attack_image = np.zeros(src_image.shape)
            opt_values1 = []
            opt_values2 = []

            for ch in range(src_image.shape[2]):
                if self.verbose is True:
                    print("Channel:", ch)
                attack_image_all, opt_values1_all, opt_values2_all = self._attack_ononedimension(src_image=src_image[:, :, ch],
                                                                              target_image=target_image[:, :, ch],
                                                                              scaler_approach=scaler_approach)
                result_attack_image[:, :, ch] = attack_image_all
                opt_values1.append(opt_values1_all)
                opt_values2.append(opt_values2_all)

            result_attack_image = result_attack_image.astype(np.uint8)
            return result_attack_image, opt_values1, opt_values2



    @abstractmethod
    def _attack_ononedimension(self, src_image: np.ndarray, target_image: np.ndarray,
               scaler_approach: ScalingApproach) \
            -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Main attack, performs the attack first in horizontal direction and then in vertical direction.
        Performs the attack on one channel.
        :param src_image:
        :param target_image:
        :param scaler_approach:
        :return: attack image, and opt values for horizontal and vertical direction.
        """
        pass

