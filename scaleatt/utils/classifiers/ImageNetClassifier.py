from abc import ABC, abstractmethod
import numpy as np
import typing


class ImageNetClassifier(ABC):

    def __init__(self):
        self.num_classes = 1000

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def preprocess_image_at_eval_time(self, image: np.ndarray):
        pass

    @abstractmethod
    def return_input_size(self) -> typing.Tuple[int, int]:
        pass

