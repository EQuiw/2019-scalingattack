import numpy as np
cimport numpy as np

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t

DTYPEint = np.int
ctypedef np.int_t DTYPEint_t


# cimport cython
# @cython.boundscheck(False) # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef random_filtering_cython(const unsigned char [:, :] att_image,
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

    np.random.seed(fr + 91*fc)

    for oj in range(xpos_length):
        pix_src_r = xpos[oj]
        pix_src_c = ypos[oj]

        ix_l = max(0, pix_src_r - bandwidthx)
        ix_r = min(pix_src_r + bandwidthx + 1, fr )
        jx_u = max(0, pix_src_c - bandwidthy)
        jx_b = min(pix_src_c + bandwidthy + 1, fc)

        for ic in range((ix_r-ix_l)*(jx_b-jx_u)):
            randpox =  np.random.randint(0, (ix_r-ix_l))
            randpoy = np.random.randint(0, (jx_b-jx_u))

            if binary_mask[ix_l:ix_r, jx_u:jx_b][randpox, randpoy] == 1:
                    continue
            filtered_attack_image[pix_src_r, pix_src_c] = att_image[ix_l:ix_r, jx_u:jx_b][randpox, randpoy]
            break

    return filtered_attack_image
