from abc import ABC
import typing
import re

from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.SuppScalingLibraries import SuppScalingLibraries

from utils.SimilarityMeasure import SimilarityMeasure


def natural_keys_natsort(text):
    """
    Used by loading attack set, so that all pickle files are sorted w.r.t to the number in string correctly.
    """

    def __atoi_natsort(text):
        return int(text) if text.isdigit() else text

    return [__atoi_natsort(c) for c in re.split(r'(\d+)', text)]


class ResultsCollection(ABC):

    def __init__(self,
                 unique_key: str,
                 scale_algorithm: SuppScalingAlgorithms,
                 scale_library: SuppScalingLibraries,
                 similarity_measurement: typing.Optional[SimilarityMeasure],
                 verbose: bool = True):


        self.unique_key: str = unique_key
        self.scale_algorithm: SuppScalingAlgorithms = scale_algorithm
        self.scale_library: SuppScalingLibraries = scale_library
        self.similarity_measurement: typing.Optional[SimilarityMeasure] = similarity_measurement
        self.verbose: bool = verbose
        self.natural_sort: bool = True # should be true, then we sort the pickle files correctly w.r.t number
