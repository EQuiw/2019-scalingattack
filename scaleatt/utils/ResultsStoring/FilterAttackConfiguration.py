import typing
import numpy as np
import os
import json

from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from usenix.utils.ResultsStoring.ResultsDefaultCollection import ResultsDefaultCollection

from usenix.utils.Configs.DatasetConfiguration import DatasetConfiguration
from usenix.utils.imagenet.load_set import load_validation_set_v4


class FilterAttackConfiguration:
    """
    Use only images that were successful w.r.t to goal A (src ~ att) and goal 2 (tar ~ out).
    """

    def __init__(self):
        a, b = FilterAttackConfiguration.get_default_values()
        self.min_scaling_factor_mapping: typing.Dict[SuppScalingAlgorithms, int] = a
        self.psnr_min = b

    @staticmethod
    def get_default_values() -> typing.Tuple[typing.Dict[SuppScalingAlgorithms, int], int]:

        min_scaling_factor_mapping = {
            SuppScalingAlgorithms.NEAREST: 3,
            SuppScalingAlgorithms.LINEAR: 4,
            SuppScalingAlgorithms.CUBIC: 4,
            SuppScalingAlgorithms.LANCZOS: 5,
        }

        psnr_min = 15

        return min_scaling_factor_mapping, psnr_min


    def filter_resultscollection(self, resultscollection: ResultsDefaultCollection) \
            -> typing.Tuple[ResultsDefaultCollection, np.ndarray]:
        scale_min = self.min_scaling_factor_mapping[resultscollection.scale_algorithm]

        range_attack_set = np.where((resultscollection.resulttable.sxsy >= scale_min) &
                                    (resultscollection.resulttable.similarity > self.psnr_min) &
                                    (resultscollection.resulttable.top5s == 1)) [0]
        print(len(range_attack_set), resultscollection.resulttable.shape[0])

        return resultscollection[range_attack_set], range_attack_set


    def export_as_json_to_file(self, target_dir: str):
        """
        Saves the class variables as json and saves them to disk
        """
        filepath = os.path.join(target_dir, "settings_filtering_attack_results.json")

        valuesx = {key: value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}

        # override dict, since enum as key cannot be exported to json.
        adapted_mapping = {}
        for key, value in self.min_scaling_factor_mapping.items():
            adapted_mapping[key.name] = value
        valuesx['min_scaling_factor_mapping'] = adapted_mapping

        with open(os.path.join(filepath), "w") as file_object:
            json.dump(valuesx, file_object)


    def filter_benign_image_set(self, rangef_benign: typing.Tuple,
                                dataset_configuration: DatasetConfiguration,
                                resultscollection: ResultsDefaultCollection) \
            -> typing.Tuple[typing.List[np.ndarray], typing.List[int]]:

        x_val_source_benign, _, _, _, _, _, _ = load_validation_set_v4(
            dataset_config=dataset_configuration, rangef=rangef_benign)

        new_sources = []
        new_sources_indices = []
        for i in reversed(range(len(x_val_source_benign))):
            # get shape
            scalex = round(x_val_source_benign[i].shape[0] / resultscollection.x_val_target[0].shape[0], 2)
            scaley = round(x_val_source_benign[i].shape[1] / resultscollection.x_val_target[0].shape[1], 2)

            if ResultsDefaultCollection.determine_scale_ratio_key(scalex=scalex, scaley=scaley) >= \
                    self.min_scaling_factor_mapping[resultscollection.scale_algorithm]:
                # use it...
                new_sources.append(x_val_source_benign[i])
                new_sources_indices.append(i + rangef_benign[0])

            if len(new_sources_indices) > resultscollection.resulttable.shape[0]:
                break

        new_sources = list(reversed(new_sources))
        new_sources_indices = list(reversed(new_sources_indices))

        return new_sources, new_sources_indices


