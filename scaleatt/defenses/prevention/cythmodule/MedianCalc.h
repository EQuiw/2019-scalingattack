#ifndef MEDIANCALC_H
#define MEDIANCALC_H

#include <vector>
#include <utility>

namespace mediancalculation {
    class MedianCalc {
        public:
//            MedianCalc();
//            ~MedianCalc();
            unsigned char get_median(std::vector<unsigned char> &vec);
            std::vector<std::pair<int, std::pair<int, int>>> argsort_matrix_abs(std::vector<std::vector<int>> &vec,
                                                                                        unsigned char target_value);
    };
}

#endif
