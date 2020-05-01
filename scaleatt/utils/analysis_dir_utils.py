import os

from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.SuppScalingLibraries import SuppScalingLibraries
from utils.Configs.OverallDatasetConfiguration import OverallDatasetConfiguration


def create_dirpath_for_saving_analysis_results(attack: bool,
                                               dataset_configuration: OverallDatasetConfiguration,
                                               scale_algorithm: SuppScalingAlgorithms,
                                               scale_library: SuppScalingLibraries,
                                               unique_key: str, directory_target: str):
    """
    As name suggests... create dir to get a unique path to save analysis results, e.g. to use for paper.
    """
    attackpath, defensepath = OverallDatasetConfiguration.get_attack_defense_paths(dataset_configuration)

    repath = attackpath if attack is True else defensepath
    save_results_dir: str = os.path.join(directory_target, os.path.basename(repath),
                                         unique_key, scale_library.name + "_" + scale_algorithm.name)

    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)

    return save_results_dir