from abc import ABC, abstractmethod
import numpy as np

from scaling.ScalingApproach import ScalingApproach


class PreventionDefense(ABC):
    """
    Defenses where we pass an image under investigation and return a 'secure' image.
    """

    def __init__(self, verbose: bool, scaler_approach: ScalingApproach):
        self.scaler_approach = scaler_approach
        self.verbose = verbose

    @abstractmethod
    def make_image_secure(self, att_image: np.ndarray) -> np.ndarray:
        """
        Return a 'secure' mage version.
        :param att_image: image under investigation.
        :return: image
        """
        pass

