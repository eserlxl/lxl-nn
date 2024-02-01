#ifndef lxl_nn_DATA_ANALYSE_H
#define lxl_nn_DATA_ANALYSE_H

#include <lxl.h>

using namespace lxl;

class Analyse {
    uzi typeIndex = 0;

    matrixString1D knownOperations = {
            "ADDITION",
            "ADDITION & SUBTRACTION",
            "MULTIPLICATION",
            "DIVISION",
            "SQRT ADDITION",
            "SINE ADDITION",
            "SINE & COSINE ADDITION",
    };

public:
    void arithmetic(matrixFloat1D &inputVec, matrixFloat1D &outputVec) const {

        switch (typeIndex) {
            // ADDITION
            case 0: {
                float sum = 0;
                for (float x : inputVec) {
                    sum += x;
                }
                outputVec = {sum};
            }
                break;
                // ADDITION & SUBTRACTION
            case 1: {
                float sum = 0, sub = 0;
                for (uzi i = 0; i < inputVec.size(); i++) {
                    sum += inputVec[i];
                    if (i % 2 == 0) {
                        sub += inputVec[i];
                    } else { sub -= inputVec[i]; }
                }
                outputVec = {sum, sub};
            }
                break;
                // MULTIPLICATION
            case 2: {
                float result = 1;
                for (float x : inputVec) {
                    result *= x;
                }
                outputVec = {result};
            }
                break;
                // DIVISON
            case 3: {
                float result = 1;
                for (float x : inputVec) {
                    if (std::fabs(x) > 1e-6)result /= x;
                    else result = INFINITY;
                }
                outputVec = {result};
            }
                break;
                // SQRT ADDITION
            case 4: {
                float sum = 0;
                for (float x : inputVec) {
                    sum += std::sqrt(x);
                }
                outputVec = {sum};
            }
                break;
                // SINE ADDITION
            case 5: {
                float sum = 0;
                for (float i : inputVec) {
                    sum += std::sin(i);
                }
                outputVec = {sum};
            }
                break;
                // SINE & COSINE ADDITION
            case 6: {
                float sum = 0;
                for (float i : inputVec) {
                    sum += std::sin(i) * std::cos(i);
                }
                outputVec = {sum};
            }
                break;
                // ADDITION
            default: {
                float sum = 0;
                for (float i : inputVec) {
                    sum += i;
                }
                outputVec = {sum};
            }
                break;
        }
    }

    void detect(matrixFloat2D &inputVec, matrixFloat2D &outputVec) {
        matrixUzi1D knownOperationsIndex(7);

        for (uzi i = 0; i < inputVec.size(); i++) {

            for (uzi j = 0; j < knownOperations.size(); j++) {
                typeIndex = j;
                matrixFloat1D tempVec;
                arithmetic(inputVec[i], tempVec);
                bool check = false;
                if (tempVec.size() == outputVec[i].size()) {
                    for (uzi k = 0; k < tempVec.size(); k++) {
                        {
                            if (!lxl::almostEqual(tempVec[k], outputVec[i][k], 1)) {
                                break;
                            } else {
                                check = true;
                            }
                        }
                    }
                }
                if (check) {
                    knownOperationsIndex[j]++;
                }
            }
        }

        typeIndex = max(knownOperationsIndex)[1];

        std::cout << "Detected data: " << knownOperations[typeIndex] << std::endl;
    }
};

#endif //lxl_nn_DATA_ANALYSE_H
