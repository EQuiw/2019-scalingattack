# distutils: language = c++

from libcpp.vector cimport vector

# ignore import error if visible, works for me, contact me if better way to do that..
from defenses.prevention.cythmodule.MedianCalc cimport MedianCalc

import numpy as np
cimport numpy as np

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t

DTYPEint = np.int
ctypedef np.int_t DTYPEint_t


# cimport cython
# @cython.boundscheck(False) # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef median_filtering_cython(const unsigned char [:, :] att_image,
                  unsigned char [:, :] filtered_attack_image,
                  const unsigned char [:, :] binary_mask,
                  long[:] xpos,
                  long[:] ypos,
                  int bandwidthx,
                  int bandwidthy):


    cdef long pix_src_r, pix_src_c
    cdef long ix_l, ix_r, jx_u, jx_b
    cdef int xpos_length = xpos.shape[0]
    cdef int randpox, randpoy
    cdef int fr = filtered_attack_image.shape[0]
    cdef int fc = filtered_attack_image.shape[1]

    cdef int ic, oj
    cdef long pi, pj
    cdef vector[unsigned char] collectedpixels

    cdef DTYPE_t res
    cdef MedianCalc medianCalc

    for oj in range(xpos_length):
        pix_src_r = xpos[oj]
        pix_src_c = ypos[oj]

        ix_l = max(0, pix_src_r - bandwidthx)
        ix_r = min(pix_src_r + bandwidthx + 1, fr )
        jx_u = max(0, pix_src_c - bandwidthy)
        jx_b = min(pix_src_c + bandwidthy + 1, fc)

        # Implement filtered_attack_image[pix_src_r, pix_src_c] = np.nanmedian(base_attack_image[ix_l:ix_r, jx_u:jx_b]):
        collectedpixels.clear()
        for pi in range(ix_l, ix_r):
            for pj in range(jx_u, jx_b):
                if binary_mask[pi, pj] != 1:
                    collectedpixels.push_back(att_image[pi, pj])

        res = medianCalc.get_median(collectedpixels)
        filtered_attack_image[pix_src_r, pix_src_c] = res

    return filtered_attack_image
