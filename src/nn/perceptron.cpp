#include <nn/perceptron.h>

void perceptron::connect(uzi outputID) {
    auto *tempLink = new synapse(id, outputID);

    Link.push_back(tempLink);
}
