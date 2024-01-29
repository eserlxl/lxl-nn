#ifndef lxl_nn_NN_SYNAPSE_H_
#define lxl_nn_NN_SYNAPSE_H_

#include <lxl.h>

using namespace lxl;

class Synapse {
public:
    uzi inputID;
    uzi outputID;
    float weight;
    float deltaWeight;

    Synapse(uzi inputID, uzi outputID) {

        this->inputID = inputID;
        this->outputID = outputID;
        weight = 0;
        deltaWeight = 0;
    }

    ~Synapse() = default;
};

#endif // lxl_nn_NN_SYNAPSE_H_
