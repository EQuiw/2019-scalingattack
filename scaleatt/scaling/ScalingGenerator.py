from scaling.CVScaler import CVScaler
from scaling.PillowScaler import PillowScaler
from scaling.TFImageScaler import TFImageScaler
from scaling.ScalingApproach import ScalingApproach

from scaling.SuppScalingLibraries import SuppScalingLibraries
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms


class ScalingGenerator:
    """
    Interface to obtain the scaling approaches for the various
    libraries and scaling algorithms.
    If you add a new scaling algorithm or library, you need to add it to this class.
    """

    def __init__(self):
        pass

    @staticmethod
    def create_scaling_approach(x_val_source_shape: tuple,
                                x_val_target_shape: tuple,
                                lib: SuppScalingLibraries,
                                alg: SuppScalingAlgorithms
                                ) -> ScalingApproach:
        """
        Creates an scaling approach for selected library (e.g. OpenCV or PIL)
        and downscaling algorithm (e.g. NEAREST or LINEAR).
        :param x_val_source_shape: source shape
        :param x_val_target_shape: target shape
        :param lib: library
        :param alg: downscaling algorithm
        :return: scaling approach
        """

        assert isinstance(alg, SuppScalingAlgorithms)
        assert isinstance(lib, SuppScalingLibraries)

        if lib == SuppScalingLibraries.CV:
            scaler_approach: ScalingApproach = CVScaler(
                algorithm=alg,
                src_image_shape=x_val_source_shape,
                target_image_shape=x_val_target_shape
            )

        elif lib == SuppScalingLibraries.TF:
            scaler_approach: ScalingApproach = TFImageScaler(
                algorithm=alg,
                src_image_shape=x_val_source_shape,
                target_image_shape=x_val_target_shape
            )
        elif lib == SuppScalingLibraries.PIL:
            scaler_approach: ScalingApproach = PillowScaler(
                algorithm=alg,
                src_image_shape=x_val_source_shape,
                target_image_shape=x_val_target_shape
            )
        else:
            raise NotImplementedError("Unknown lib passed")

        return scaler_approach


    @staticmethod
    def get_all_lib_alg_combinations():
        """
        Returns a dict with all possible library + algorithm combinations.
        """
        scale_algorithm_list = {
            SuppScalingLibraries.CV: [SuppScalingAlgorithms.NEAREST, SuppScalingAlgorithms.LINEAR,
                                      SuppScalingAlgorithms.CUBIC, SuppScalingAlgorithms.LANCZOS,
                                       SuppScalingAlgorithms.AREA],
            SuppScalingLibraries.PIL: [SuppScalingAlgorithms.NEAREST, SuppScalingAlgorithms.LINEAR,
                                      SuppScalingAlgorithms.CUBIC, SuppScalingAlgorithms.LANCZOS,
                                       SuppScalingAlgorithms.AREA],
            SuppScalingLibraries.TF: [SuppScalingAlgorithms.NEAREST, SuppScalingAlgorithms.LINEAR,
                                      SuppScalingAlgorithms.CUBIC,
                                       SuppScalingAlgorithms.AREA],
        }
        return scale_algorithm_list


    @staticmethod
    def check_valid_lib_alg_input(lib: SuppScalingLibraries,
                                alg: SuppScalingAlgorithms) -> bool:
        """
        Checks if passed algorithm is implemented for passed library.
        For instance, TF may not have Lanczos.
        TODO we may use get_all_lib_alg_combinations for that?
        :param lib: scaling lib.
        :param alg: scaling alg.
        :return: true if valid.
        """
        try:
            ScalingGenerator.create_scaling_approach(
                x_val_source_shape=(100,100),
                x_val_target_shape=(100,100),
                lib=lib, alg=alg
            )
            return True
        except NotImplementedError:
            return False