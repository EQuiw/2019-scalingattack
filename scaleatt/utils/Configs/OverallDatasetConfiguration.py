import os
import enum
import json
from abc import ABC, abstractmethod

from utils.classifiers.ClassiferNames import ClassifierNames


class OverallDatasetConfiguration(ABC):
    """
    Dataset configuration.
    """

    def __init__(self, directory_files_base: str, dataset_id: str, classifier_name: ClassifierNames):
        """
        Dataset configuration
        :param directory_files_base: path to overall directory where results will be saved.
        :param dataset_id: a unique name for the dataset constellation
        :param classifier_name: the used classifier (actually the images do not depend on the used classifier,
        however, as we use a classifier to check if the success of the image-scaling attack w.r.t goal O1,
        we check here already that source and target do not belong to the same class w.r.t predictions..
        """

        self.dataset_id = dataset_id
        self.classifier_name: ClassifierNames = classifier_name
        self.directory_files_base: str = directory_files_base
        self.directory_dataset: str = os.path.join(directory_files_base, dataset_id + "_" + classifier_name.name)


    def export_as_json_to_file(self):
        """
        Saves the class variables as json and saves them to disk, so that we can load the Configuration
        from this json again. The advantage over pickle is that we can change the Configuration Class
        by adding e.g. variables. Then the json file would not contain have them and show a warning.
        Pickle would return an error if the loaded class does not correspond to the source code here.
        """
        filepath = os.path.join(self.directory_dataset, "settings_dataset.json")

        valuesx = {key: value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}

        # convert enum to its name
        for k, v in valuesx.items():
            if isinstance(v, enum.Enum):
                valuesx[k] = v.name

        with open(os.path.join(filepath), "w") as file_object:
            json.dump(valuesx, file_object)

    @staticmethod
    @abstractmethod
    def load_from_dict(directory_files_base: str, dataset_id: str, classifier_name: ClassifierNames):
        pass


    @staticmethod
    def get_attack_defense_paths(dat: 'OverallDatasetConfiguration'):

        directory_files_base = dat.directory_files_base
        dataset_id = dat.dataset_id
        classifier_name = dat.classifier_name

        # Where attack images should be stored
        directory_savedfiles_attackbase: str = os.path.join(directory_files_base, "attacks",
                                                            "attack_" + dataset_id + "_" + classifier_name.name)
        # Where defense images should be stored
        directory_savedfiles_defensebase: str = os.path.join(directory_files_base, "defenses",
                                                             "defense_" + dataset_id + "_" + classifier_name.name)

        return directory_savedfiles_attackbase, directory_savedfiles_defensebase