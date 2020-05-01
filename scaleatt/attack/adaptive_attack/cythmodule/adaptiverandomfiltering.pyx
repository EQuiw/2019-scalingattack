# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.utility cimport pair as cpp_pair
from libcpp cimport bool

# ignore import error if visible, works for me, contact me if better way to do that..
from defenses.prevention.cythmodule.MedianCalc cimport MedianCalc

import numpy as np
cimport numpy as np

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t

DTYPEint = np.int
ctypedef np.int_t DTYPEint_t




cpdef vector[cpp_pair[int, cpp_pair[int, int]]] argsort_indices(const unsigned char [:, :] input_image,
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



# cimport cython
# @cython.boundscheck(False) # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef adaptive_attack_random_filtering_cython(const unsigned char [:, :] att_image,
                  unsigned char [:, :] filtered_attack_image,
                  const unsigned char [:, :] binary_mask,
                  long[:] xpos,
                  long[:] ypos,
                  int bandwidthx,
                  int bandwidthy,
                  float allowed_ratio_of_change):
    #assert att_image.dtype == DTYPE and filtered_attack_image.dtype == DTYPE and binary_mask.dtype == DTYPE
    #assert xpos.dtype == DTYPEint and ypos.dtype == DTYPEint


    cdef long pix_src_r, pix_src_c
    cdef long ix_l, ix_r, jx_u, jx_b
    cdef int xpos_length = xpos.shape[0]
    cdef int randpox, randpoy
    cdef int fr = filtered_attack_image.shape[0]
    cdef int fc = filtered_attack_image.shape[1]

    cdef int ic, oj, cbr, cbj
    cdef long pi, pj
    # cdef vector[unsigned char] collectedpixels

    cdef unsigned char target_value
    cdef bool increase

    cdef vector[float] l0_changes
    cdef int changes = 0
    cdef int possible_changes

    cdef int no_success = 0
    cdef bool is_success
    cdef float no_success_score

    # cdef const float [:, :] cur_block_result
    cdef const unsigned char [:, :] cur_binary_mask

    cdef vector[cpp_pair[int, cpp_pair[int, int]]] sorted_indices_block
    cdef int diff
    cdef cpp_pair[int, cpp_pair[int, int]] vec_index
    cdef cpp_pair[int, int] pair_index
    cdef int block_r, block_c
    cdef unsigned int block_i

    for oj in range(xpos_length):

        changes = 0
        pix_src_r = xpos[oj]
        pix_src_c = ypos[oj]
        target_value = att_image[pix_src_r, pix_src_c]

        ix_l = max(0, pix_src_r - bandwidthx)
        ix_r = min(pix_src_r + bandwidthx + 1, fr )
        jx_u = max(0, pix_src_c - bandwidthy)
        jx_b = min(pix_src_c + bandwidthy + 1, fc)

        cur_binary_mask = binary_mask[ix_l:ix_r, jx_u:jx_b]

        # cur_block_result = filtered_attack_image[ix_l:ix_r, jx_u:jx_b]
        # cur_block_result = float_att_image[ix_l:ix_r, jx_u:jx_b]

        # dist = np.abs(cur_block_result - target_value)
        # sorted_indices_block = np.dstack(np.unravel_index(np.argsort(dist.ravel()), dist.shape))[0]

        sorted_indices_block = argsort_indices(att_image[ix_l:ix_r, jx_u:jx_b], target_value)

        possible_changes = get_possible_changes(binary_mask, ix_l, ix_r, jx_u, jx_b)

        is_success = False
        for block_i in range(sorted_indices_block.size()):
            # unpack vector to get sorted indices:
            vec_index = sorted_indices_block[block_i]
            diff = vec_index.first
            pair_index = vec_index.second
            block_r = pair_index.first
            block_c = pair_index.second

            # check if we can do changes and if so do it:
            if (changes / possible_changes) >= allowed_ratio_of_change:
                is_success = True
                break

            if cur_binary_mask[block_r, block_c] != 1:
                # filtered_attack_image[ix_l:ix_r, jx_u:jx_b][block_r, block_c] = target_value # same same
                filtered_attack_image[ix_l + block_r, jx_u + block_c] = target_value
                changes += 1

        if not is_success:
            no_success += 1
        l0_changes.push_back(changes / possible_changes)

    no_success_score = no_success/len(xpos)
    return filtered_attack_image, l0_changes, no_success_score
