#include "nn/neuron.h"

void Neuron::connect(uzi outputID) {
    auto *tempLink = new Synapse(id, outputID);

    Link.push_back(tempLink);
}
