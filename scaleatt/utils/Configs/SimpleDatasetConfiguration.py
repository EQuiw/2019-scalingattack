import typing

from utils.classifiers.ClassiferNames import ClassifierNames
from utils.Configs.OverallDatasetConfiguration import OverallDatasetConfiguration


class SimpleDatasetConfiguration(OverallDatasetConfiguration):
    """
    As DatasetConfiguration, but for datasets where we can load the images directly
    from library, as for CIFAR or MNIST.
    """

    def __init__(self, directory_files_base: str, dataset_id: str, classifier_name: ClassifierNames,
                 intervals_eval: typing.Optional[list] = None):

        super().__init__(directory_files_base=directory_files_base,
                         dataset_id=dataset_id, classifier_name=classifier_name)

        self.intervals_eval: typing.Optional[list] = intervals_eval
        self.directory_dataset = None # no path to dataset as directly loaded from library


    @staticmethod
    def load_from_dict(directory_files_base: str, dataset_id: str,
                       classifier_name: ClassifierNames):
        raise NotImplementedError("No need to save and load config for this simple dataset config")


    # @Overwrite
    def export_as_json_to_file(self):
        raise NotImplementedError("No need to save and load config for this simple dataset config")

