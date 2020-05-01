import numpy as np
import typing

from defenses.prevention.RandomFilteringDefense import RandomFilteringDefense
from attack.adaptive_attack.AdaptiveAttack import AdaptiveAttackOnAttackImage
from scaling.ScalingApproach import ScalingApproach

from attack.adaptive_attack.cythmodule.adaptiverandomfiltering import adaptive_attack_random_filtering_cython



class AdaptiveRandomAttack(AdaptiveAttackOnAttackImage):
    """
    Adaptive attack from paper to mislead random-filter-based defense (defense 2 in paper).
    """

    def __init__(self,
                 verbose: bool,
                 scaler_approach: ScalingApproach,
                 randomfilteringdefense: RandomFilteringDefense,
                 choose_only_unused_pixels_in_overlapping_case: bool,
                 allowed_ratio_of_change: float,
                 usecython: bool
                 ):
        """

        :param verbose: print debug messages
        :param scaler_approach: scaler approach
        :param randomfilteringdefense: randomfilteringdefense
        :param choose_only_unused_pixels_in_overlapping_case: if true, we try to change only pixels
        that were not changed before. if false, we ignore previously set values in overlapping windows.
        Preliminary results show that false leads to better results visually..
        :param allowed_ratio_of_change:  in %, the number of pixels that can be changed.
        :param usecython: use cython-based attack..
        """
        super().__init__(verbose, scaler_approach)
        self.randomfilteringdefense = randomfilteringdefense
        # self.eps = 3

        self.choose_only_unused_pixels_in_overlapping_case = choose_only_unused_pixels_in_overlapping_case

        self.last_run_changed_pixels: typing.List[typing.List[float]] = []
        self.last_run_nosuccess: typing.List[float] = []
        self.allowed_ratio_of_change: float = allowed_ratio_of_change
        self.usecython = usecython

        if self.choose_only_unused_pixels_in_overlapping_case == True and self.usecython == True:
            raise Exception("choose_only_unused_pixels_in_overlapping_case not implemented in cython, yet")



    def get_stats_last_run(self) -> typing.Tuple[typing.List[typing.List[float]] , typing.List[float]]:
        return self.last_run_changed_pixels, self.last_run_nosuccess



    # @Overwrite
    def counter_attack(self, att_image: np.ndarray) -> np.ndarray:

        # I. get binary mask
        # todo a cleaner way would be to use the binary mask from medianfilteringdefense. make class stateful then.
        dir_attack_image = self.randomfilteringdefense.fourierpeakmatrixcollector.get(
            scaler_approach=self.randomfilteringdefense.scaler_approach)
        binary_mask_indices = np.where(dir_attack_image != 255)
        binary_mask = np.zeros((self.randomfilteringdefense.scaler_approach.cl_matrix.shape[1],
                                self.randomfilteringdefense.scaler_approach.cr_matrix.shape[0]))
        binary_mask[binary_mask_indices] = 1

        # II. go over each channel if necessary
        if len(att_image.shape) == 2:
            if not self.usecython:
                r = self.__apply_attack(att_image=att_image, binary_mask=binary_mask)
            else:
                r = self.__apply_attack_cython(att_image=att_image, binary_mask=binary_mask)
            return r.astype(np.uint8)
        else:
            filtered_att_image = np.zeros(att_image.shape)
            for ch in range(att_image.shape[2]):
                if self.verbose is True:
                    print("Channel:", ch)

                if not self.usecython:
                    re = self.__apply_attack(att_image=att_image[:, :, ch], binary_mask=binary_mask)
                else:
                    re = self.__apply_attack_cython(att_image=att_image[:, :, ch], binary_mask=binary_mask)

                filtered_att_image[:, :, ch] = re
            return filtered_att_image.astype(np.uint8)


    def __apply_attack(self, att_image, binary_mask):
        filtered_attack_image = np.copy(att_image)
        base_attack_image = np.copy(att_image).astype(np.float32)
        positions = np.where(binary_mask == 1)

        if self.choose_only_unused_pixels_in_overlapping_case is True:
            assert np.any(np.isnan(base_attack_image)) == False
            base_marked_attack_image = np.zeros(base_attack_image.shape)
            base_marked_attack_image[positions] = 1

        # apply filter
        xpos = positions[0]
        ypos = positions[1]

        no_success: int = 0 # counter for non-successful windows
        l0_changes = [] # count the number of changed pixels per window
        for pix_src_r, pix_src_c in zip(xpos, ypos):
            target_value = att_image[pix_src_r, pix_src_c]  # the target value

            ix_l = max(0, pix_src_r - self.randomfilteringdefense.bandwidth[0])
            ix_r = min(pix_src_r + self.randomfilteringdefense.bandwidth[0] + 1, filtered_attack_image.shape[0])
            jx_u = max(0, pix_src_c - self.randomfilteringdefense.bandwidth[1])
            jx_b = min(pix_src_c + self.randomfilteringdefense.bandwidth[1] + 1, filtered_attack_image.shape[1])

            # find x% of pixels that are closest to target value and then change them.
            cur_block = base_attack_image[ix_l:ix_r, jx_u:jx_b]
            cur_block_result = filtered_attack_image[ix_l:ix_r, jx_u:jx_b]
            if self.choose_only_unused_pixels_in_overlapping_case is True:
                cur_marked_block = base_marked_attack_image[ix_l:ix_r, jx_u:jx_b]

            # We use cur_block and not cur_block_result (even if overlapping), which gave better results.
            # Consider data types: Use cur_block that was converted to float32, otherwise integer overflow.
            dist = np.abs(cur_block - target_value)
            # dist = np.abs(cur_block_result.astype(np.float32) - target_value) #
            sorted_indices_block = np.dstack(np.unravel_index(np.argsort(dist.ravel()), dist.shape))[0]


            if self.choose_only_unused_pixels_in_overlapping_case is True:
                possible_changes = np.sum(base_marked_attack_image[ix_l:ix_r, jx_u:jx_b] == 0)
            else:
                possible_changes = np.sum(binary_mask[ix_l:ix_r, jx_u:jx_b] == 0)

            changes = 0
            success = False
            for block_r, block_c in sorted_indices_block:

                if (changes/possible_changes) >= self.allowed_ratio_of_change:
                    success = True
                    break

                if binary_mask[ix_l:ix_r, jx_u:jx_b][block_r, block_c] == 1:
                    continue

                if self.choose_only_unused_pixels_in_overlapping_case is True:
                    if cur_marked_block[block_r, block_c] == 1:
                        continue
                    else:
                        cur_marked_block[block_r, block_c] = 1

                cur_block_result[block_r, block_c] = target_value
                changes += 1

            if success is False:
                no_success += 1
            l0_changes.append(changes / possible_changes)


        if self.verbose:
            print("No success: {} ({}%)".format(no_success, no_success / len(xpos)))

        self.last_run_nosuccess.append(no_success / len(xpos))
        self.last_run_changed_pixels.append(l0_changes)

        return filtered_attack_image


    def __apply_attack_cython(self, att_image: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
        """
        This is a cython wrapper that calls the respective cython function. Much faster than the Python version.
        :param att_image: image under investigation
        :param binary_mask: binary mask: pixels that are considered
        :return: filtered image
        """

        filtered_attack_image = np.copy(att_image)
        positions = np.where(binary_mask == 1)
        xpos = positions[0]
        ypos = positions[1]

        res, l0_changes, no_success_score = adaptive_attack_random_filtering_cython(att_image, filtered_attack_image,
                                                                  binary_mask.astype(np.uint8), xpos, ypos,
                                                                  self.randomfilteringdefense.bandwidth[0],
                                                                  self.randomfilteringdefense.bandwidth[1],
                                                                  # self.eps,
                                                                  self.allowed_ratio_of_change)

        self.last_run_changed_pixels.append(l0_changes)
        self.last_run_nosuccess.append(no_success_score)

        return np.array(res)  # cython returns memoryview..

