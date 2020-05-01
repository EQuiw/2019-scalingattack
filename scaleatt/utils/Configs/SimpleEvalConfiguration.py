from scaling.SuppScalingLibraries import SuppScalingLibraries
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from utils.Configs.OverallEvalConfiguration import OverallEvalConfiguration


class SimpleEvalConfiguration(OverallEvalConfiguration):

    def __init__(self, dnn_name, dataset_id, unique_key, directory_load_dataset, directory_saved_files_base,
                 eps_attack: list,
                 scale_library: SuppScalingLibraries,
                 scale_algorithm: SuppScalingAlgorithms):
        """
        Simple variant of Configuration for attack + defense
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

        super().__init__(dnn_name=dnn_name, dataset_id=dataset_id,
                         unique_key=unique_key, directory_load_dataset=directory_load_dataset,
                         directory_saved_files_base=directory_saved_files_base,
                         eps_attack=eps_attack, scale_library=scale_library, scale_algorithm=scale_algorithm)

    @staticmethod
    def load_from_dict(directory_result: str, directory_savedfiles_attackbase: str,
                       directory_load_dataset_expected: str):
        raise NotImplementedError()


    def erase_values_when_merging(self):
        """
        If we merge two Configuration objects, then set those values to None that make no sense here anymore.
        """
        self.directory_saved_files = None

