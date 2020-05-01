### Defense 2, Adaptive Attack (Section 5.5)###
# We evaluate our defenses against an adaptive adversary who is aware of the defenses
# and adapts her attack accordingly.

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
from attack.adaptive_attack.AdaptiveAttackPreventionGenerator import AdaptiveAttackPreventionGenerator
from attack.adaptive_attack.AdaptiveAttack import AdaptiveAttackOnAttackImage


########################## Set up ##########################
# First question: Have you run the cython initialization code? If not, set here False, then we use
# the python code to run the defense, but the cython version is much faster.
usecythonifavailable: bool = True

# Parameters
scaling_algorithm: SuppScalingAlgorithms = SuppScalingAlgorithms.NEAREST
scaling_library: SuppScalingLibraries = SuppScalingLibraries.TF
args_bandwidthfactor: int = 2

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

fourierpeakmatrixcollector: FourierPeakMatrixCollector = FourierPeakMatrixCollector(
    method=PeakMatrixMethod.optimization, scale_library=scaling_library, scale_algorithm=scaling_algorithm
)

########################## Median Filter Defense #########################

args_prevention_type = PreventionTypeDefense.medianfiltering
preventiondefense: PreventionDefense = PreventionDefenseGenerator.create_prevention_defense(
            defense_type=args_prevention_type, scaler_approach=scaler_approach,
            fourierpeakmatrixcollector=fourierpeakmatrixcollector,
            bandwidth=args_bandwidthfactor, verbose_flag=False, usecythonifavailable=usecythonifavailable
        )


## New! Init and run the adaptive attack
args_allowed_changes: int = 50 # the percentage of pixels that can be modified in each block.
adaptiveattack: AdaptiveAttackOnAttackImage = AdaptiveAttackPreventionGenerator.create_adaptive_attack(
            defense_type=args_prevention_type, scaler_approach=scaler_approach,
            preventiondefense=preventiondefense,
            verbose_flag=False, usecythonifavailable=usecythonifavailable,
            choose_only_unused_pixels_in_overlapping_case=False,
            allowed_changes=args_allowed_changes/100
        )

adaptiveattackimage = adaptiveattack.counter_attack(att_image=result_attack_image)

# now apply our median defense
filtered_attack_image = preventiondefense.make_image_secure(att_image=adaptiveattackimage)
downscl_filtered_att_image = scaler_approach.scale_image(xin=filtered_attack_image)
# downscl_Unfiltered_att_image = scaler_approach.scale_image(xin=result_attack_image)

# Let's analyze the two goals of the attack
# - O1: the downscaled output of the attack image should correspond to the target image
# - Observation: Both images correspond to each other, so the adpative attack worked
plot_images_in_actual_size([downscl_filtered_att_image, tar_image_example], ["Attack", "Target"],  rows=1)

# Now let's analyze the other goal of the attack
# - O2: the attack image should look like the source image before downscaling
# - Observation: The modifications are clearly visible in the attack image, so the attack did not work!
# In order to obtain the target image as downscaled result, too many modifications are necessary to circumvent
#  the median filter.
plot_images_in_actual_size([adaptiveattackimage, src_image_example], ["Attack", "Source"],  rows=1)



########################## Random Filter Defense #########################
# Now let's analyze the random filter.

args_prevention_type = PreventionTypeDefense.randomfiltering
preventiondefense: PreventionDefense = PreventionDefenseGenerator.create_prevention_defense(
            defense_type=args_prevention_type, scaler_approach=scaler_approach,
            fourierpeakmatrixcollector=fourierpeakmatrixcollector,
            bandwidth=args_bandwidthfactor, verbose_flag=False, usecythonifavailable=usecythonifavailable
        )

args_allowed_changes: int = 75 # the percentage of pixels that can be modified in each block.
adaptiveattack: AdaptiveAttackOnAttackImage = AdaptiveAttackPreventionGenerator.create_adaptive_attack(
            defense_type=args_prevention_type, scaler_approach=scaler_approach,
            preventiondefense=preventiondefense,
            verbose_flag=False, usecythonifavailable=usecythonifavailable,
            choose_only_unused_pixels_in_overlapping_case=False,
            allowed_changes=args_allowed_changes/100
        )

adaptiveattackimage = adaptiveattack.counter_attack(att_image=result_attack_image)

# now apply our random defense
filtered_attack_image = preventiondefense.make_image_secure(att_image=adaptiveattackimage)
downscl_filtered_att_image = scaler_approach.scale_image(xin=filtered_attack_image)
downscl_Unfiltered_att_image = scaler_approach.scale_image(xin=result_attack_image)

# Let's analyze the two goals of the attack
# - O1: the output of the attack image should correspond to the target image
# - I've added here the downscaled image from the adaptive attack in two variants
#   - if we use it without applying the random filter before downscaling = downscl_Unfiltered_att_image
#   - if we use it by applying the random filter before = downscl_filtered_att_image
# - Observation: The random filter destroys the downscaled output a little, since the adaptive attack
#   stops when we changed 75% of each block. But the random filter might take another pixel, so that
#   we get an overlay. However, if we directly used the downscaled output of the adaptive attack,
#   'downscl_Unfiltered_att_image', we can see that the attack worked.
plot_images_in_actual_size([downscl_Unfiltered_att_image, downscl_filtered_att_image,
                            tar_image_example], ["Output", "Out. after filtering", "Target"],  rows=1)


# Now let's analyze the other goal of the attack
# - O2: the attack image should look like the source image before downscaling
# - Observation: The modifications are clearly visible in the attack image, so the attack did not work!
# In order to obtain the target image as downscaled result, too many modifications are necessary to circumvent
#  the random filter.
plot_images_in_actual_size([adaptiveattackimage, src_image_example], ["Attack", "Source"],  rows=1)