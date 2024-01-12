#include <nn.h>

void nn::initValues() {
    for (uzi i = 0; i < P.size(); i++) {
        for (uzi j = 0; j < P[i].size(); j++) {
            P[i][j]->value = network[i][j];
        }
    }
}

void nn::setIO(std::vector<float> input, std::vector<float> output) {
    if (input.size() != model[0]) {
        std::cout << "Improper network design according to inputs!" << std::endl;
        exit(-1);
    }
    if (output.size() != model[outputIndex]) {
        std::cout << "Improper network design according to outputs!" << std::endl;
        exit(-1);
    }
    for (uzi i = 0; i < model[0]; i++) {
        network[0][i] = input[i];
        P[0][i]->value = input[i];
    }
    for (uzi i = 0; i < model[outputIndex]; i++) {
        network[outputIndex][i] = output[i];
    }
}

void nn::initEtaAlpha() {
    eta = 10;
    alpha = -0.002;
}