import os
import json
import enum
from abc import ABC, abstractmethod

from scaling.SuppScalingLibraries import SuppScalingLibraries
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms


class OverallEvalConfiguration(ABC):

    def __init__(self, dnn_name, dataset_id, unique_key, directory_load_dataset, directory_saved_files_base,
                 eps_attack: list,
                 scale_library: SuppScalingLibraries,
                 scale_algorithm: SuppScalingAlgorithms):
        """
        Configuration for attack + defense
        :param dnn_name: name of DNN, e.g. VGG19 or InceptionV3
        :param dataset_id: dataset id
        :param unique_key: unique key for evaluation
        :param directory_load_dataset: path to pickle files from 1_createdataset steps or "None" if directly loaded
        from lib, e.g. if we use CIFAR10
        :param directory_saved_files_base: directory where results will be saved (new directory is created)
        :param eps_attack: epsilon parameter of scaling attack
        :param scale_library: scaling library
        :param scale_algorithm: scaling algorithm
        """

        self.dnn_name = dnn_name
        self.dataset_id = dataset_id
        self.unique_key = unique_key
        self.directory_load_dataset = directory_load_dataset

        assert type(eps_attack) == list
        self.eps_attack: list = eps_attack

        # assert that all enum parameters obtain a respective enum, and not e.g. an int!
        assert isinstance(scale_algorithm, SuppScalingAlgorithms) and isinstance(scale_library, SuppScalingLibraries)
        self.scale_algorithm = scale_algorithm
        self.scale_library = scale_library

        self._base_dir = directory_saved_files_base
        overall_dir_results: str = OverallEvalConfiguration.get_results_paths(
            basepath=self._base_dir,
            unique_key=self.unique_key,
            scale_library=self.scale_library,
            scale_algorithm=self.scale_algorithm)

        self.directory_saved_files = os.path.join(overall_dir_results, "_".join(["results_imgs"]))


    @staticmethod
    def get_results_paths(basepath: str, unique_key: str, scale_library: SuppScalingLibraries,
                 scale_algorithm: SuppScalingAlgorithms) -> str:
        return os.path.join(basepath, unique_key, scale_library.name + "_" + scale_algorithm.name)


    def export_as_json_to_file(self):
        """
        Saves the class variables as json and saves them to disk, so that we can load the Configuration
        from this json again. The advantage over pickle is that we can change the Configuration Class
        by adding e.g. variables. Then the json file would not contain have them and show a warning.
        Pickle would return an error if the loaded class does not correspond to the source code here.
        """
        filepath = os.path.join(self.directory_saved_files, "settings_dict.json")

        valuesx = {key: value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}

        # convert enum to its name
        for k, v in valuesx.items():
            if isinstance(v, enum.Enum):
                valuesx[k] = v.name

        with open(os.path.join(filepath), "w") as file_object:
            json.dump(valuesx, file_object)


    @staticmethod
    @abstractmethod
    def load_from_dict(directory_result: str, directory_savedfiles_attackbase: str,
                       directory_load_dataset_expected: str):
        pass


    def __str__(self):
        valuesx = {key: value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}
        return str(valuesx)


    @abstractmethod
    def erase_values_when_merging(self):
        """
        If we merge two Configuration objects, then set those values to None that make no sense here anymore.
        """
        pass


    @staticmethod
    def determine_if_can_be_merged(configuration1: 'OverallEvalConfiguration', configuration2: 'OverallEvalConfiguration'):
        """
        If we merge two Configuration objects, then check that both confugrations can be merged.
        :param configuration1:
        :param configuration2:
        """
        assert configuration1.scale_algorithm == configuration2.scale_algorithm
        assert configuration1.scale_library == configuration2.scale_library

        assert len(configuration1.eps_attack) == len(configuration2.eps_attack)
        for ep, ep2 in zip(configuration1.eps_attack, configuration2.eps_attack):
            assert ep == ep2
        # assert configuration1.eps_attack == configuration2.eps_attack
        assert configuration1.dnn_name == configuration2.dnn_name
        assert configuration1.dataset_id == configuration2.dataset_id
        assert configuration1.unique_key == configuration2.unique_key
        assert configuration1._base_dir == configuration2._base_dir