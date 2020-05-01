import typing

from defenses.prevention.MedianFilteringDefense import MedianFilteringDefense
from defenses.prevention.RandomFilteringDefense import RandomFilteringDefense
from defenses.prevention.PreventionDefense import PreventionDefense
from defenses.prevention.PreventionDefenseType import PreventionTypeDefense

from scaling.ScalingApproach import ScalingApproach
from defenses.detection.fourier.FourierPeakMatrixCollector import FourierPeakMatrixCollector



class PreventionDefenseGenerator:
    """
    Generator for various prevention defenses.
    """

    @staticmethod
    def create_prevention_defense(defense_type: PreventionTypeDefense, verbose_flag: bool,
                                  scaler_approach: ScalingApproach,
                                  fourierpeakmatrixcollector: FourierPeakMatrixCollector,
                                  bandwidth: typing.Optional[int],
                                  usecythonifavailable: bool) -> PreventionDefense:
        """
        Creates a specific prevention defense.
        """

        if defense_type == PreventionTypeDefense.medianfiltering:
            preventiondefense: PreventionDefense = MedianFilteringDefense(verbose=verbose_flag,
                                                                                scaler_approach=scaler_approach,
                                                                                fourierpeakmatrixcollector=fourierpeakmatrixcollector,
                                                                                bandwidth=bandwidth, usecython=usecythonifavailable)
        elif defense_type == PreventionTypeDefense.randomfiltering:
            preventiondefense: PreventionDefense = RandomFilteringDefense(verbose=verbose_flag,
                                                                               scaler_approach=scaler_approach,
                                                                               fourierpeakmatrixcollector=fourierpeakmatrixcollector,
                                                                               bandwidth=bandwidth, usecython=usecythonifavailable)
        else:
            raise NotImplementedError()

        return preventiondefense

