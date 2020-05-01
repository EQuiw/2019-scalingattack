import enum
import numpy as np
import pickle
import os
import copy

from scaling.ScalingApproach import ScalingApproach
from scaling.ScalingGenerator import ScalingGenerator
from attack.QuadrScaleAttack import QuadraticScaleAttack
from attack.ScaleAttackStrategy import ScaleAttackStrategy
from attack.direct_attacks.DirectNearestScaleAttack import DirectNearestScaleAttack

from scaling.SuppScalingLibraries import SuppScalingLibraries
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms


class PeakMatrixMethod(enum.Enum):
    """
    Method to determine the considered pixels in source image.
    """

    # If we know the scaling implementation, we can directy access the pixels that are used by algorithm.
    # This is the optimal way, and we implemented this for OpenCV+Nearest scaling. However, do not use
    # it so far, as somehow due to rounding issues, it is not completely equivalent!!
    direct_nearest_cv = 1
    # We use an image-scaling attack to obtain the pixels that are used by the scaling algorithm.
    # It is the slowest option, but in this way, we only get the pixels that the attack algorithm is really
    # using. This option was used for our USENIX Sec.'20 paper.
    optimization = 2
    # This is an approximation that provides good results for the scaling algorithms IF the scaling ratio is large.
    # However, it can mark too many pixels to be used by a scaling algorithm. Use this option only for quick tests
    # and for your final evaluation, use optimization.
    cl_cr_approx = 3



class FourierPeakMatrixCollector:
    """
    Allows us to obtain a matrix where all pixels that are processed by a scaling algorithm are marked.
    Remind that not all pixels are necessarily used, and not all are equally used.

    Memorizes the matrices for a specific instance of library + algorithm, so that we can get
    the matrices faster for different scaling ratios.
    """

    def __init__(self, method: PeakMatrixMethod,
                 scale_algorithm: SuppScalingAlgorithms,
                 scale_library: SuppScalingLibraries):
        """
        Init this class.
        We need scale-algorithm and library to link this class to a unique setting of scaler approach.
        The source- and target shapes do not matter, but we should not mix up different peak matrices
        for varying libraries or algorithms, because each peak matrix depends on the library + algorithm.
        :param method:
        :param scale_algorithm:
        :param scale_library:
        """

        # create hash map that stores for each source-target shape pair
        # a respective attack image
        self.att_imgs: dict = {}
        self.method: PeakMatrixMethod = method

        # we check that all following approaches have the same setup (library and algorithm).
        # create dummy scaler approach to this end.
        self.first_scaling_approach: ScalingApproach = ScalingGenerator.create_scaling_approach(
            x_val_source_shape=(100, 100),
            x_val_target_shape=(50, 50),
            lib=scale_library,
            alg=scale_algorithm)

        self.scale_lib = scale_library
        self.scale_alg = scale_algorithm



    def get(self,  scaler_approach: ScalingApproach):

        # A. Consistency check: is scaler_approach the same as before?
        if self.first_scaling_approach is None:
            self.first_scaling_approach = scaler_approach
        else:
            if self.first_scaling_approach.get_unique_approach_identifier() \
                    != scaler_approach.get_unique_approach_identifier():
                raise Exception("Setup in passed scaling approach is different to the first passed approach here")

        # B. Get data:
        ckey: str = FourierPeakMatrixCollector.__get_key(scaler_approach=scaler_approach)

        if ckey in self.att_imgs:
            return self.att_imgs[ckey]
        else:
            return self.__get_peak_matrix(scaler_approach=scaler_approach)


    def __get_peak_matrix(self, scaler_approach: ScalingApproach):

        # get shape information
        src_shape0 = scaler_approach.cl_matrix.shape[1]
        tar_shape0 = scaler_approach.cl_matrix.shape[0]

        src_shape1 = scaler_approach.cr_matrix.shape[0]
        tar_shape1 = scaler_approach.cr_matrix.shape[1]

        src_image = np.ones((src_shape0, src_shape1), dtype=np.uint8) * 255
        tar_image = np.ones((tar_shape0, tar_shape1), dtype=np.uint8) * 100

        # get attack image
        # create map: we can compute the frequencies that we expect in attack image as peaks
        if self.method == PeakMatrixMethod.direct_nearest_cv:

            direct_scale_attack: DirectNearestScaleAttack = DirectNearestScaleAttack(verbose=False)
            att_img, _, _ = direct_scale_attack.attack(src_image=src_image,
                                                       target_image=tar_image,
                                                       scaler_approach=scaler_approach)

        elif self.method == PeakMatrixMethod.optimization:

            scale_att: ScaleAttackStrategy = QuadraticScaleAttack(eps=1, verbose=False)
            att_img, _, _ = scale_att.attack(src_image=src_image,
                                             target_image=tar_image,
                                             scaler_approach=scaler_approach)

        elif self.method == PeakMatrixMethod.cl_cr_approx:
            att_img = FourierPeakMatrixCollector.approx_scale_coeffs(
                scaler_approach=scaler_approach,
                src_image=src_image
            )
        else:
            raise NotImplementedError()

        # save and return
        ckey: str = FourierPeakMatrixCollector.__get_key(scaler_approach=scaler_approach)
        self.att_imgs[ckey] = att_img
        return att_img


    @staticmethod
    def __get_key(scaler_approach: ScalingApproach):
        src_shape0 = scaler_approach.cl_matrix.shape[1]
        tar_shape0 = scaler_approach.cl_matrix.shape[0]

        src_shape1 = scaler_approach.cr_matrix.shape[0]
        tar_shape1 = scaler_approach.cr_matrix.shape[1]

        return "_".join([str(src_shape0), str(src_shape1), str(tar_shape0), str(tar_shape1)])


    @staticmethod
    def approx_scale_coeffs(scaler_approach: ScalingApproach, src_image: np.ndarray):
        aj = scaler_approach.cl_matrix.T
        bj = scaler_approach.cr_matrix.T

        bbj = np.tile(bj.sum(axis=0),
                      (src_image.shape[0], 1))
        aaj = np.repeat(aj.sum(axis=1), src_image.shape[1]).reshape(src_image.shape[0], src_image.shape[1])

        return aaj * bbj



    ### ******* Load and save this class ******* ###

    @staticmethod
    def save_to_disk(collector: 'FourierPeakMatrixCollector',
                     directory_filtered_dataset: str,
                     dataset_id: str,
                     overwrite: bool = False) -> None:
        """
        Save this class as pickle file to passed directory. This method SHOULD NOT be used
        if save_to_disk could be called in parallel!
        :param collector: FourierPeakMatrixCollector
        :param directory_filtered_dataset: directory where the pickle file should be stored.
        :param dataset_id: dataset id
        :param overwrite: overwrite existing file.
        """

        target_dir = os.path.join(directory_filtered_dataset, "fourierpeakmatrixcollector", dataset_id)

        assert collector.first_scaling_approach is not None
        # name_ = collector.first_scaling_approach.get_unique_approach_identifier() + "_" + collector.method.name
        name_ = collector.scale_lib.name + "_" + collector.scale_alg.name + "_" + collector.method.name

        if not os.path.exists(target_dir):
            print("Create target dir:", target_dir)
            os.makedirs(target_dir)

        file_coll = os.path.join(target_dir, name_ + "_peakmatrixcollector.pck")
        if os.path.exists(file_coll) and overwrite is False:
            print("Target File already exists")
            return


        # if no file exists, we get an empty collector. So no need to consider the special case if no file exists.
        loadedpeakmatrixcollector = FourierPeakMatrixCollector.load_from_disk_or_create(
                        method=collector.method, directory_filtered_dataset=directory_filtered_dataset,
                        dataset_id=dataset_id, scale_algorithm=collector.scale_alg, scale_library=collector.scale_lib
                    )

        # merge both collectors into a new collector
        newcollector: 'FourierPeakMatrixCollector' = FourierPeakMatrixCollector.merge(
                        collector1=collector, collector2=loadedpeakmatrixcollector
                    )

        # now save it
        with open(file_coll, 'wb') as file_object:
            pickle.dump(newcollector, file_object)

        print("Could save collector")


    @staticmethod
    def load_from_disk_or_create(method: 'PeakMatrixMethod',
                                 directory_filtered_dataset: str,
                                 dataset_id: str,
                                 scale_algorithm: SuppScalingAlgorithms,
                                 scale_library: SuppScalingLibraries):

        target_dir = os.path.join(directory_filtered_dataset, "fourierpeakmatrixcollector", dataset_id)

        # create dummy scaler approach
        scaler_approach: ScalingApproach = ScalingGenerator.create_scaling_approach(
            x_val_source_shape=(100, 100),
            x_val_target_shape=(50, 50),
            lib=scale_library,
            alg=scale_algorithm)

        # load scaler approach if it exists
        # name_ = scaler_approach.get_unique_approach_identifier() + "_" + method.name
        name_ = scale_library.name + "_" + scale_algorithm.name + "_" + method.name
        file_coll = os.path.join(target_dir, name_ + "_peakmatrixcollector.pck")

        if os.path.exists(target_dir) and os.path.exists(file_coll):
            print("Load from existing file")

            with open(file_coll, 'rb') as file_object:
                fourierpeakmatrixcollector = pickle.load(file_object)
            if fourierpeakmatrixcollector.first_scaling_approach.get_unique_approach_identifier() \
                    != scaler_approach.get_unique_approach_identifier():
                raise Exception("Setup in load scaling approach is different to the expected approach")

        else:
            print("Create new PeakMatrixCollector")

            fourierpeakmatrixcollector: FourierPeakMatrixCollector = FourierPeakMatrixCollector(
                method=method, scale_algorithm=scale_algorithm, scale_library=scale_library
            )

        return fourierpeakmatrixcollector


    @staticmethod
    def merge(collector1: 'FourierPeakMatrixCollector', collector2: 'FourierPeakMatrixCollector') \
            -> 'FourierPeakMatrixCollector':

        collector = copy.deepcopy(collector1)

        for key in collector2.att_imgs:
            if key not in collector.att_imgs:
                collector.att_imgs[key] = collector2.att_imgs[key]

        return collector



