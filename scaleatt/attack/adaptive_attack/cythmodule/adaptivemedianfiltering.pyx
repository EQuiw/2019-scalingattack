# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.utility cimport pair as cpp_pair
from libcpp cimport bool

cdef extern from "<algorithm>" namespace "std":
    void reverse[iter](iter first, iter second)



# ignore import error if visible, works for me, contact me if better way to do that..
from defenses.prevention.cythmodule.MedianCalc cimport MedianCalc

import numpy as np
cimport numpy as np

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t

DTYPEint = np.int
ctypedef np.int_t DTYPEint_t



cdef unsigned char get_median(const unsigned char [:, :] &att_image,
                const unsigned char [:, :] &binary_mask,
                long ix_l, long ix_r, long jx_u, long jx_b):
    cdef MedianCalc medianCalc
    cdef vector[unsigned char] collectedpixels
    cdef unsigned char cur_median

    for pi in range(ix_l, ix_r):
            for pj in range(jx_u, jx_b):
                if binary_mask[pi, pj] != 1:
                    collectedpixels.push_back(att_image[pi, pj])

    cur_median = medianCalc.get_median(collectedpixels)
    return cur_median


cdef vector[cpp_pair[int, cpp_pair[int, int]]] argsort_indices(const unsigned char [:, :] input_image,
                                                                const unsigned char target_value):
    cdef MedianCalc medianCalc
    cdef vector[cpp_pair[int, cpp_pair[int, int]]] return_val

    # TODO check if better way to convert memoryview to c++ vector without creating python array in the middle
    return_val = medianCalc.argsort_matrix_abs(np.array(input_image, dtype=DTYPEint), target_value)
    return return_val



cdef int get_possible_changes(const unsigned char [:, :] &binary_mask,
                long ix_l, long ix_r, long jx_u, long jx_b):

    cdef int possible_changes = 0

    for pi in range(ix_l, ix_r):
            for pj in range(jx_u, jx_b):
                if binary_mask[pi, pj] != 1:
                    possible_changes += 1

    return possible_changes


# cdef set_median_like_values(
#         unsigned char [:, :] &filtered_attack_image,
#         const unsigned char [:, :] &binary_mask,
#         long ix_l, long ix_r, long jx_u, long jx_b,
#         unsigned char cur_median, bool increase,
#         unsigned char target_value, int* changes, int possible_changes, float allowed_ratio_of_change):
#
#     for pi in range(ix_l, ix_r):
#         for pj in range(jx_u, jx_b):
#             if binary_mask[pi, pj] != 1:
#                 if cur_median == filtered_attack_image[pi, pj]:
#
#                     if (changes[0] / possible_changes) > allowed_ratio_of_change:
#                         return
#
#                     if increase:
#                         if get_median(filtered_attack_image, binary_mask, ix_l, ix_r, jx_u, jx_b) < target_value:
#                             filtered_attack_image[pi, pj] = target_value
#                             changes[0] +=1
#                         else:
#                             return
#                     if not increase:
#                         if get_median(filtered_attack_image, binary_mask, ix_l, ix_r, jx_u, jx_b) > target_value:
#                             filtered_attack_image[pi, pj] = target_value
#                             changes[0] +=1
#                         else:
#                             return


cdef vector[cpp_pair[int, cpp_pair[int, int]]] take_closest_values(
        bool &increase,
        unsigned char &target_value,
        unsigned char [:,:] &filtered_attack_image,
        const unsigned char [:, :] &binary_mask):

    cdef vector[cpp_pair[int, cpp_pair[int, int]]] sorted_indices_block
    cdef vector[cpp_pair[int, cpp_pair[int, int]]] result_indices_block

    cdef cpp_pair[int, cpp_pair[int, int]] vec_index
    cdef cpp_pair[int, int] pair_index
    cdef int block_r, block_c
    cdef unsigned int block_i = 0


    # get closest values
    sorted_indices_block = argsort_indices(filtered_attack_image, 0)

    # now get sorting for iteration later: if increase, we'd like to iterate from the back to the front,
    #   if decrease, normal order. Thus, if increase is True, we reverse the order.
    if increase:
        reverse(sorted_indices_block.begin(), sorted_indices_block.end())

    for block_i in range(0, sorted_indices_block.size()):
        # unpack vector to get sorted indices:
        vec_index = sorted_indices_block[block_i]
        pair_index = vec_index.second
        block_r = pair_index.first
        block_c = pair_index.second

        # check that current value is not considered by downscaling algorithm
        if binary_mask[block_r, block_c] != 1:
            # check if increase is True and value < target_value:
            if increase and filtered_attack_image[block_r, block_c] < target_value:
                result_indices_block.push_back(vec_index)
            if not increase and filtered_attack_image[block_r, block_c] > target_value:
                result_indices_block.push_back(vec_index)

    return result_indices_block


# cimport cython
# @cython.boundscheck(False) # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef adaptive_attack_median_filtering_cython(const unsigned char [:, :] att_image,
                  unsigned char [:, :] filtered_attack_image,
                  const unsigned char [:, :] binary_mask,
                  long[:] xpos,
                  long[:] ypos,
                  int bandwidthx,
                  int bandwidthy,
                  int eps,
                  float allowed_ratio_of_change):
    #assert att_image.dtype == DTYPE and filtered_attack_image.dtype == DTYPE and binary_mask.dtype == DTYPE
    #assert xpos.dtype == DTYPEint and ypos.dtype == DTYPEint



    cdef long pix_src_r, pix_src_c
    cdef long ix_l, ix_r, jx_u, jx_b
    cdef int xpos_length = xpos.shape[0]
    cdef int randpox, randpoy
    cdef int fr = filtered_attack_image.shape[0]
    cdef int fc = filtered_attack_image.shape[1]

    cdef int ic, oj
    cdef long pi, pj
    # cdef vector[unsigned char] collectedpixels

    cdef unsigned char target_value
    cdef bool increase

    cdef DTYPE_t cur_median
    # cdef MedianCalc medianCalc

    cdef vector[float] l0_changes
    cdef int no_success = 0
    cdef bool is_success
    cdef int changes = 0
    cdef int possible_changes


    cdef vector[cpp_pair[int, cpp_pair[int, int]]] block_pixels
    cdef cpp_pair[int, cpp_pair[int, int]] vec_index
    cdef cpp_pair[int, int] pair_index
    cdef int block_r, block_c
    cdef unsigned int block_i
    cdef unsigned char na_med
    cdef float no_success_score


    for oj in range(xpos_length):
        changes = 0
        pix_src_r = xpos[oj]
        pix_src_c = ypos[oj]
        target_value = att_image[pix_src_r, pix_src_c]

        ix_l = max(0, pix_src_r - bandwidthx)
        ix_r = min(pix_src_r + bandwidthx + 1, fr )
        jx_u = max(0, pix_src_c - bandwidthy)
        jx_b = min(pix_src_c + bandwidthy + 1, fc)

        cur_median = get_median(filtered_attack_image, binary_mask, ix_l, ix_r, jx_u, jx_b)
        # cur_median = get_median(att_image, binary_mask, ix_l, ix_r, jx_u, jx_b)
        possible_changes = get_possible_changes(binary_mask, ix_l, ix_r, jx_u, jx_b)

        #
        if abs(target_value - cur_median) < eps:
            continue # no integer overflow here due to cython

        increase = (target_value > cur_median)

        block_pixels = take_closest_values(increase, target_value, filtered_attack_image[ix_l:ix_r, jx_u:jx_b],
                                           binary_mask[ix_l:ix_r, jx_u:jx_b])

        is_success = False
        for block_i in range(block_pixels.size()):
            # unpack vector to get sorted indices:
            vec_index = block_pixels[block_i]
            pair_index = vec_index.second
            block_r = pair_index.first
            block_c = pair_index.second

            na_med = get_median(filtered_attack_image, binary_mask, ix_l, ix_r, jx_u, jx_b)
            if increase and na_med >= target_value:
                is_success = True
                break
            elif not increase and na_med <= target_value:
                is_success = True
                break

            # check if we can do changes and if so do it:
            if (changes / possible_changes) >= allowed_ratio_of_change:
                break

            if binary_mask[ix_l + block_r, jx_u + block_c] != 1:
                filtered_attack_image[ix_l + block_r, jx_u + block_c] = target_value
                changes += 1

        if not is_success:
            no_success += 1
        l0_changes.push_back(changes / possible_changes)

        # Old Strategy:
        # for pi in range(ix_l, ix_r):
        #     for pj in range(jx_u, jx_b):
        #         if binary_mask[pi, pj] != 1:
        #
        #             if (changes / possible_changes) > allowed_ratio_of_change:
        #                 break
        #
        #             if increase == True and cur_median < filtered_attack_image[pi, pj] < target_value:
        #                 filtered_attack_image[pi, pj] = target_value
        #                 changes += 1
        #             if increase == False and cur_median > filtered_attack_image[pi, pj] > target_value:
        #                 filtered_attack_image[pi, pj] = target_value
        #                 changes += 1
        #
        # # quick and dirty here.
        # # problem: if we have multiple pixels == median, then we cannot set all these values to new value.
        # # currently, we set all median-like values until new median ~ target value
        # set_median_like_values(filtered_attack_image, binary_mask, ix_l, ix_r, jx_u, jx_b,
        #                                                cur_median, increase, target_value, &changes,
        #                        possible_changes, allowed_ratio_of_change)

    no_success_score = no_success/len(xpos)
    return filtered_attack_image, l0_changes, no_success_score
