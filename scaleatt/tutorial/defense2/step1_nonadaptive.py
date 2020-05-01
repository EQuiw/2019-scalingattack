### Defense 2, Non-Adaptive Attack (Section 5.4) ###
# We evaluate our novel defenses for reconstructing images. We combine here
# the selective median and random filter with a vulnerable scaling algorithm.

from utils.plot_image_utils import plot_images_in_actual_size

from scaling.ScalingGenerator import ScalingGenerator
from scaling.SuppScalingLibraries import SuppScalingLibraries
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.ScalingApproach import ScalingApproach
from attack.QuadrScaleAttack import QuadraticScaleAttack
from attack.ScaleAttackStrategy import ScaleAttackStrategy
from utils.load_image_data import load_image_examples

from defenses.prevention.PreventionDefenseType import PreventionTypeDefense
from defenses.prevention.PreventionDefenseGenerator import PreventionDefenseGenerator
from defenses.prevention.PreventionDefense import PreventionDefense

from defenses.detection.fourier.FourierPeakMatrixCollector import FourierPeakMatrixCollector, PeakMatrixMethod

########################## Set up ##########################
# First question: Have you run the cython initialization code? If not, set here False, then we use
# the python code to run the defense, but the cython version is much faster.
usecythonifavailable: bool = True

# Parameters
scaling_algorithm: SuppScalingAlgorithms = SuppScalingAlgorithms.LINEAR
scaling_library: SuppScalingLibraries = SuppScalingLibraries.CV

########################## Load image ##########################

src_image_example, tar_image_example = load_image_examples(img_src=2, plot_loaded=False)

assert ScalingGenerator.check_valid_lib_alg_input(lib=scaling_library, alg=scaling_algorithm) is True

########################## Attack #########################
## Now perform attack before we apply our median filter as defense

scaler_approach: ScalingApproach = ScalingGenerator.create_scaling_approach(
    x_val_source_shape=src_image_example.shape,
    x_val_target_shape=tar_image_example.shape,
    lib=scaling_library,
    alg=scaling_algorithm
)

scale_att: ScaleAttackStrategy = QuadraticScaleAttack(eps=1, verbose=True)
result_attack_image, _, _ = scale_att.attack(src_image=src_image_example,
                                             target_image=tar_image_example,
                                             scaler_approach=scaler_approach)

########################## Median Filter Defense #########################
# settings for the defense
args_bandwidthfactor = 2

# we need a matrix that tells us which pixels could be manipulated by an attack - Object can be reused.
fourierpeakmatrixcollector: FourierPeakMatrixCollector = FourierPeakMatrixCollector(
    method=PeakMatrixMethod.optimization, scale_library=scaling_library, scale_algorithm=scaling_algorithm
)

# specify the prevention defense, we currently have medianfiltering and randomfiltering.
args_prevention_type = PreventionTypeDefense.medianfiltering

# Init the defense.
preventiondefense: PreventionDefense = PreventionDefenseGenerator.create_prevention_defense(
    defense_type=args_prevention_type, scaler_approach=scaler_approach,
    fourierpeakmatrixcollector=fourierpeakmatrixcollector,
    bandwidth=args_bandwidthfactor, verbose_flag=False, usecythonifavailable=usecythonifavailable
)

# this function might take longer because we need to initialize fourierpeakmatrixcollector. If you run the code again,
# the runtime of the defense should be negligible. Please don't do any runtime benchmarking experiments in Python,
# and use our C++ code for that (it is more optimized and shows how a library would use the defense).
filtered_attack_image = preventiondefense.make_image_secure(att_image=result_attack_image)

downscl_filtered_att_image = scaler_approach.scale_image(xin=filtered_attack_image)

# Now let's compare if the defense was able to restore the input to the actual source image before any attack
plot_images_in_actual_size([result_attack_image, filtered_attack_image, src_image_example],
                           ["Attack image before defens", "Attack image after defense", "Actual original source image"],
                           rows=1)

# Now compare the downscaled result. We can see that the attack was not successful.
# (D): The first image is the downscaled version of the repaired attack image;
# (O): the 2nd image is the actual goal of the attacker.
plot_images_in_actual_size([downscl_filtered_att_image, tar_image_example],
                           ["D", "O"], rows=1)

########################## Random Filter Defense #########################
# Now repeat the defense with the random filter.
args_prevention_type_random = PreventionTypeDefense.randomfiltering

preventiondefense_random: PreventionDefense = PreventionDefenseGenerator.create_prevention_defense(
    defense_type=args_prevention_type_random, scaler_approach=scaler_approach,
    fourierpeakmatrixcollector=fourierpeakmatrixcollector,
    bandwidth=args_bandwidthfactor, verbose_flag=False, usecythonifavailable=usecythonifavailable
)

filtered_attack_image_random = preventiondefense_random.make_image_secure(att_image=result_attack_image)

# Observation: The random filter also prevents the attack, but the median filter gives better results visually.
plot_images_in_actual_size([result_attack_image, filtered_attack_image, filtered_attack_image_random],
                           ["Attack image before defens", "Attack after median filter", "Attack after random filter"],
                           rows=1)


