### Step 3, Adaptive Attack ###
# Next, we examine our adaptive attack strategy against Area scaling. The original attack from Xiao et al.
# is not applicable, so that we use a new method.
# We omit the adaptive strategies against Pillow and the selective source image scenario here

from utils.plot_image_utils import plot_images_in_actual_size

from scaling.ScalingGenerator import ScalingGenerator
from scaling.SuppScalingLibraries import SuppScalingLibraries
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.ScalingApproach import ScalingApproach
from utils.load_image_data import load_image_examples, load_color_image_example_from_disk

import attack.area_attack.rescale_area_utils
from attack.area_attack.area_straight_scale_attack import area_straight_scale_attack
# from attack.area_attack.area_scale_nonintegerborders import area_scale_attack_nonintegerborders
from attack.area_attack.area_scale_integerborders import area_scale_attack_integerborders
from attack.area_attack.AreaNormEnumType import AreaNormEnumType

########################## Load image ##########################
# We first load an image pair that serves as source- and target image. We use the cat and coffee image
# from our paper. Note that we can use any image pair, but consider our root-cause analysis section
# for the conditions of a successful attack (scaling algorithm and scaling ratio).

src_image_example, tar_image_example = load_image_examples(img_src=2)

# If you want to use an image from your disk,
# you can also use 'load_color_image_example_from_disk' to load two images from your disk directly.

##########################   Attack   ##########################

# L0, L1 or L2 norm for attack against Area. L1 and L2 give very similar results.
# Depending on the norm, we will have to use another attack strategy (see our paper).
area_attack_norm: AreaNormEnumType = AreaNormEnumType.L0

# Permutation: If true, we distribute the L0 changes randomly in each block that area scaling considers,
# otherwise we always start in the upper left corner. After preliminary experiments, we set it to False as default value.
args_usepermutation: bool = False

# We first scale the source image to a multiple of the target image shape, so that the scaling ratio becomes
# an integer. This makes the attack easier; if we have no integer ratio, then we have overlapping blocks
# in area scaling, and we need to optimize multiple blocks at the same time. In the worst case, we cannot find
# a solution. By using an integer as scaling ratio , we evaluate the harder case from the defender side.
src_image_example = attack.area_attack.rescale_area_utils.rescale_to_integer(
    noninteger=False, src_image=src_image_example, tar_image_shape=tar_image_example.shape, row=0
)

scaler_approach: ScalingApproach = ScalingGenerator.create_scaling_approach(
    x_val_source_shape=src_image_example.shape,
    x_val_target_shape=tar_image_example.shape,
    lib=SuppScalingLibraries.CV,  # do not change here to other lib. with area
    alg=SuppScalingAlgorithms.AREA)

# Perfom the attack.
if area_attack_norm == AreaNormEnumType.L1 or area_attack_norm == AreaNormEnumType.L2:
    result_attack_image = area_scale_attack_integerborders(
        tar_image=tar_image_example, src_img=src_image_example,
        eps=3, blockwise=True, verbose=False, attack_norm=area_attack_norm)
    changed_pixels_str = None
elif area_attack_norm == AreaNormEnumType.L0:
    result_attack_image, changed_pixels_str = area_straight_scale_attack(tar_image=tar_image_example,
                                                                         src_img=src_image_example,
                                                                         verbose=False,
                                                                         permutation=args_usepermutation)
else:
    raise NotImplementedError()

result_output_image = scaler_approach.scale_image(xin=result_attack_image)

# Now we will observe that
#   a) the attack image is not really nice. We had to change many pixels to obtain the wanted result after area scaling,
#       so that the resulting attack image has clear traces of the target image.
#   b) the output image of the attack image corresponds to the target image (meaning that our adaptive attack
#   against area scaling worked).
plot_images_in_actual_size(imgs=[src_image_example, result_attack_image], titles=["Source", "Attack"], rows=1)
plot_images_in_actual_size(imgs=[tar_image_example, result_output_image], titles=["Target", "Output"], rows=1)

# Finally, set 'area_attack_norm' to L1 in line 32, and re-run the experiments.
# Check how the result looks like by using our L1-based adaptive attack against Area.
# In this case, the attack image will look like the target image before scaling!
# Consider that the attack image should look like the source image for a successful attack.
# Thus, an image-scaling attack was not possible here against area scaling.
