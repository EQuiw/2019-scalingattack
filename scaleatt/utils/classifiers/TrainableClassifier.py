from abc import ABC, abstractmethod
import numpy as np
import typing

class TrainableClassifier(ABC):

    def __init__(self):
        self.num_classes = self.get_number_classes()

    @abstractmethod
    def get_number_classes(self) -> int:
        pass

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def preprocess_image_at_eval_time(self, image: np.ndarray):
        pass

    @abstractmethod
    def return_input_size(self) -> typing.Tuple[int, int]:
        pass


    ### Methods for learning ###

    @abstractmethod
    def train_model(self, train_x, train_y, val_x, val_y, epochs: int):
        pass

    @abstractmethod
    def evaluate(self, x_test, y_test):
        pass

    @abstractmethod
    def predict_classes(self, x_test):
        pass

    @abstractmethod
    def save_model(self, path: str):
        pass

    @abstractmethod
    def load_model(self, path: str):
        pass
