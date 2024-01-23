#include <nn/neuron.h>

void neuron::connect(uzi outputID) {
    auto *tempLink = new synapse(id, outputID);

    Link.push_back(tempLink);
}
