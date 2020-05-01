from abc import ABC, abstractmethod
import numpy as np
import typing

from scaling.ScalingApproach import ScalingApproach


class DetectionDefense(ABC):
    """
    Defenses where we pass an image under investigation and >detect< if it is an attack image or not.
    """

    def __init__(self, verbose: bool, scaler_approach: ScalingApproach):
        self.scaler_approach = scaler_approach
        self.verbose = verbose

    @abstractmethod
    def detect_attack(self, att_image: np.ndarray) -> float:
        """
        Detect attack
        :param att_image:
        :return: a score w.r.t attack
        """
        pass

