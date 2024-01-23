#ifndef lxl_nn_NN_NEURON_H_
#define lxl_nn_NN_NEURON_H_

#include <vector>
#include <nn/synapse.h>

class neuron {
public:
    uzi id;
    float value;

    std::vector<synapse *> Link;

    void connect(uzi outputID);

    template<typename Float>
    neuron(uzi perceptronNo, Float initialValue) {
        id = perceptronNo;

        value = initialValue;
    }

    ~neuron() {
        for (auto &link : Link) {
            delete (link);
        }
        Link.clear();
    }
};

#endif // lxl_nn_NN_NEURON_H_
