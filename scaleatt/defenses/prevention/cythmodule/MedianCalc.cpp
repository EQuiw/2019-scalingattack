#include "MedianCalc.h"
#include<algorithm>
#include <math.h>
#include <stdexcept>

namespace mediancalculation {

    unsigned char MedianCalc::get_median(std::vector<unsigned char> &vec) {

    if(vec.empty()){
        throw std::invalid_argument( "received empty vec" );
    }

    if (vec.size() % 2 == 0) {
//        const auto median_it1 = vec.begin() + vec.size() / 2 - 1;
//        const auto median_it2 = vec.begin() + vec.size() / 2;
//
//        std::nth_element(vec.begin(), median_it1 , vec.end());
//        const auto e1 = *median_it1;
//
//        std::nth_element(vec.begin(), median_it2 , vec.end());
//        const auto e2 = *median_it2;
//
////        return (e1 + e2) / 2; Instead of an average, we need an integer.
//        unsigned char ret = floor((e1+e2)/2);
//        return ret;

        // We avoid taking the average of two values, and simply use the left one of the two, as the average
        // might allow to move the median with less changes if the adversary uses an adaptive attack == open question.
        const auto median_it1 = vec.begin() + vec.size() / 2 - 1;
        std::nth_element(vec.begin(), median_it1 , vec.end());
        const auto e1 = *median_it1;
        unsigned char ret = e1;
        return ret;

    } else {
        const auto median_it = vec.begin() + vec.size() / 2;
        std::nth_element(vec.begin(), median_it , vec.end());
        return *median_it;
    }

    }

    std::vector<std::pair<int, std::pair<int, int>>> MedianCalc::argsort_matrix_abs(
                                                                                    std::vector<std::vector<int>> &vec,
                                                                                    unsigned char target_value) {

        // Vector to store elements with indices, we use a pair of pairs, since I haven't found tuple for cython.
        std::vector<std::pair<int, std::pair< int, int>>> vp;

        // Inserting elements in vector
        for (unsigned int i = 0; i < vec.size(); i++) {
            for(unsigned int j = 0; j < vec[0].size(); j++) {
                int diff = abs(vec[i][j] - target_value);
                vp.push_back(std::make_pair( diff, std::make_pair(i, j)));
            }
        }

        // Sorting pair vector, important that the pixel value is at the first location, then sort will sort
        // pairs wrt first value.
        sort(vp.begin(), vp.end());

        return vp;
    }
}