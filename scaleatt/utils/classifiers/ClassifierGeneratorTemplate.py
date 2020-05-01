from abc import ABC, abstractmethod
from utils.classifiers.ClassiferNames import ClassifierNames


class ClassifierGeneratorTemplate(ABC):

    @staticmethod
    @abstractmethod
    def getimageclassifier(name: ClassifierNames):
        pass
