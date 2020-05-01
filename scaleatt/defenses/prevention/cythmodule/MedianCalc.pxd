from libcpp.vector cimport vector
from libcpp.utility cimport pair as cpp_pair

cdef extern from "MedianCalc.cpp":
    pass

# Declare the class with cdef
cdef extern from "MedianCalc.h" namespace "mediancalculation":
    cdef cppclass MedianCalc:
        unsigned char get_median(vector[unsigned char] &vec) except +
        vector[cpp_pair[int, cpp_pair[int, int]]] argsort_matrix_abs(vector[vector[int]] &vec,
                                                                     unsigned char target_value) except +
