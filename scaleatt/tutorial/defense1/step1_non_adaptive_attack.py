### Step 1, Non-Adaptive Attack ###
# In the first experiment, we will examine the robustness of existing scaling algorithms
# against image-scaling attacks. To this end, we use the default image-scaling attack (non-adaptive).

from utils.plot_image_utils import plot_images_in_actual_size

from scaling.ScalingGenerator import ScalingGenerator
from scaling.SuppScalingLibraries import SuppScalingLibraries
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.ScalingApproach import ScalingApproach
from attack.QuadrScaleAttack import QuadraticScaleAttack
from attack.ScaleAttackStrategy import ScaleAttackStrategy
from utils.load_image_data import load_image_examples, load_color_image_example_from_disk



########################## Load image ##########################
# We first load an image pair that serves as source- and target image. We use a cat and coffee image.
# For this tutorial, I've reduced the image size of the cat to (60, 90) to speed up the computation.
# Note that our images from ImageNet for the evaluation are much larger.

src_image_example, tar_image_example = load_image_examples(img_src=2)
# ensure that images are integers in the range [0,255] and use numpy.uint8 as data type.


# Moreover, we can use any image pair, but consider our root-cause analysis section
# for the conditions of a successful attack (scaling algorithm and scaling ratio).
# If you want to use an image from your disk,
# you can use our function 'load_color_image_example_from_disk' to load two images from your disk directly.

##########################   Attack   ##########################

# 1. First, choose the scaling algorithm and library to be used.
# Let's use the nearest scaling algorithm which is often the default algorithm.
scaling_algorithm: SuppScalingAlgorithms = SuppScalingAlgorithms.NEAREST
# We support OpenCV, Pillow or TensorFlow (consider that Pillow has a 'secure' scaling behaviour for
# Linear and Cubic, see Sec. 4.2)
scaling_library: SuppScalingLibraries = SuppScalingLibraries.CV

# check that library has implemented the algorithm. TensorFlow, for instance, has no Lanczos..
assert ScalingGenerator.check_valid_lib_alg_input(lib=scaling_library, alg=scaling_algorithm) is True



# 2. We need to generate a scaling object. It will provide everything that we need to scale images,
# and it holds the scaling matrices that our optimization approach needs.
scaler_approach: ScalingApproach = ScalingGenerator.create_scaling_approach(
    x_val_source_shape=src_image_example.shape,
    x_val_target_shape=tar_image_example.shape,
    lib=scaling_library,
    alg=scaling_algorithm
)


# 3. Next, we run the scaling attack that gives us the attack image;
# By choosing an epsilon of 1, the maximum difference between the downscaled attack image and our target image should be 1.
# The attack may take some minutes.
scale_att: ScaleAttackStrategy = QuadraticScaleAttack(eps=1, verbose=True)

result_attack_image, _, _ = scale_att.attack(src_image=src_image_example,
                                             target_image=tar_image_example,
                                             scaler_approach=scaler_approach)

# Now we use the attack image and scale it down, as we would do in a real machine learning pipeline.
result_output_image = scaler_approach.scale_image(xin=result_attack_image)


# 4. Now plot the images; We use 'plot_images_in_actual_size' to plot images in their correct resolution.
# Otherwise, if the plot method itself uses some scaling algorithm, we may see or see no artifacts.

# Observation:
# A succesful attack is characterized by two goals (see paper, Section 2.2.1)
# O1: The downscaled attack image should correspond to the target image (look at the plot, it does)
plot_images_in_actual_size(imgs=[tar_image_example, result_output_image], titles=["Target", "Output"], rows=1)

# O2: The attack image should correspond to the original source image (look at the plot, it does)
plot_images_in_actual_size(imgs=[src_image_example, result_attack_image], titles=["Source", "Attack"], rows=1)
