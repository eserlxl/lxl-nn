#ifndef lxl_nn_NN_NEURON_H_
#define lxl_nn_NN_NEURON_H_

#include <vector>
#include <lxl.h>
#include "synapse.h"

using namespace lxl;
typedef std::vector<Synapse *> matrixSynapse1D;

class Neuron {
public:
    uzi id;
    float value;

    matrixSynapse1D Link;

    void connect(uzi outputID);

    template<typename Float>
    Neuron(uzi neuronID, Float initialValue) {
        id = neuronID;

        value = initialValue;
    }

    ~Neuron() {
        for (auto &link : Link) {
            safeDelete(link);
        }
        Link.clear();
    }
};

#endif // lxl_nn_NN_NEURON_H_
