### Step 2, Non-Adaptive Attack ###
# Now we look closer at the impact of the scaling ratio and kernel size (see Section 3.3, and e.g. Figure 7)

from utils.plot_image_utils import plot_images_in_actual_size, plot_images1_actual_size

from scaling.ScalingGenerator import ScalingGenerator
from scaling.SuppScalingLibraries import SuppScalingLibraries
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.ScalingApproach import ScalingApproach
from attack.QuadrScaleAttack import QuadraticScaleAttack
from attack.ScaleAttackStrategy import ScaleAttackStrategy
from utils.load_image_data import load_image_examples
from scaling.scale_utils import scale_cv2


########################## Load image ##########################

src_image_example, tar_image_example = load_image_examples(img_src=2)

##########################   Attack   ##########################
scaling_library: SuppScalingLibraries = SuppScalingLibraries.CV

# We will now vary the scaling algorithm (and in this way the kernel size), and we will vary the scaling ratio.

###### A. Vary the kernel size ######

# we save all attack images and output images after downscaling to compare the impact
results_attacks = []
results_outputs = []

for current_scaling_algorithm in [SuppScalingAlgorithms.NEAREST, SuppScalingAlgorithms.LINEAR,
                                  SuppScalingAlgorithms.CUBIC]:
    scaler_approach: ScalingApproach = ScalingGenerator.create_scaling_approach(
        x_val_source_shape=src_image_example.shape,
        x_val_target_shape=tar_image_example.shape,
        lib=scaling_library,
        alg=current_scaling_algorithm
    )

    scale_att: ScaleAttackStrategy = QuadraticScaleAttack(eps=1, verbose=True)

    result_attack_image, _, _ = scale_att.attack(src_image=src_image_example,
                                                 target_image=tar_image_example,
                                                 scaler_approach=scaler_approach)

    result_output_image = scaler_approach.scale_image(xin=result_attack_image)
    results_attacks.append(result_attack_image)
    results_outputs.append(result_output_image)
#

# The outputs are identical to the target image, so the attack succeeded always (goal O1)
plot_images_in_actual_size(imgs=results_outputs, titles=["Nearest o=1", "Linear o=2", "Cubic o=4"], rows=1)

# But with respect to goal O2, a closer analysis reveals when the attack is successful.
# Let's plot the impact of different kernel widths (as given by the scaling algorithm). The parameter o corresponds
# to the kernel width from paper (see Table 3 and Eq. 9)
plot_images_in_actual_size(imgs=results_attacks, titles=["Nearest o=1", "Linear o=2", "Cubic o=4"], rows=1)

# Observation: The changes on the second image are slightly stronger. This is because we have a kernel width of 2,
# so that we need two pixels for each output pixel in order to manipulate the downscaling approach.
# Moreover, the third image is similar to the second image. Consider Figure 6 from our paper
# and our root-cause analysis from Section 3.3:  The kernel is larger, but the weight is still centered
# around two pixels. As a result, linear and cubic lead to similar attack images.



###### B. Vary the scaling ratio ######

# We gradually increase the size of the target image so that the scaling ratio decreases

results_attacks = []

scaling_algorithm = SuppScalingAlgorithms.NEAREST
for scaleratiofactor in [1, 1.5, 2, 4]:
    current_tar_image_example = scale_cv2(xin=tar_image_example,
                                          trows=int(tar_image_example.shape[0] * scaleratiofactor),
                                          tcols=int(tar_image_example.shape[1] * scaleratiofactor))

    scaler_approach: ScalingApproach = ScalingGenerator.create_scaling_approach(
        x_val_source_shape=src_image_example.shape,
        x_val_target_shape=current_tar_image_example.shape,
        lib=scaling_library,
        alg=scaling_algorithm
    )

    scale_att: ScaleAttackStrategy = QuadraticScaleAttack(eps=1, verbose=True)

    result_attack_image, _, _ = scale_att.attack(src_image=src_image_example,
                                                 target_image=current_tar_image_example,
                                                 scaler_approach=scaler_approach)

    result_output_image = scaler_approach.scale_image(xin=result_attack_image)
    results_attacks.append(result_attack_image)

# Observation: The smaller the scaling ratio, the more visible the attack is.
# This is because more pixels are modified w.r.t to the whole image (see Sec. 3.3)
plot_images_in_actual_size(imgs=results_attacks, titles=["1", "1.5", "2", "4"], rows=1)
