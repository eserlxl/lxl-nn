#ifndef lxl_nn_NN_PERCEPTRON_H_
#define lxl_nn_NN_PERCEPTRON_H_

#include <vector>
#include <nn/synapse.h>

class perceptron {
public:
    uzi id;
    float value;

    std::vector<synapse *> Link;

    void connect(uzi outputID);

    template<typename Float>
    perceptron(uzi perceptronNo, Float initialValue) {
        id = perceptronNo;

        value = initialValue;
    }

    ~perceptron() {
        for (auto &link : Link) {
            delete(link);
        }
        Link.clear();
    }
};

#endif // lxl_nn_NN_PERCEPTRON_H_
