#ifndef lxl_nn_NN_NEURON_H_
#define lxl_nn_NN_NEURON_H_

#include <vector>
#include <lxl.h>
#include "synapse.h"

using namespace lxl;

class Neuron {
public:
    uzi id;
    float value;

    std::vector<Synapse *> Link;

    void connect(uzi outputID);

    template<typename Float>
    Neuron(uzi perceptronNo, Float initialValue) {
        id = perceptronNo;

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
