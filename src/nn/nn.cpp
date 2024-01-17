#include <nn.h>

void nn::create() {
    uzi id = 0;
    for (uzi i : model) {
        std::vector<perceptron *> tempVec;

        for (uzi j = 0; j < i; j++) {
            auto *temp = new perceptron(id++, 0.f);
            tempVec.push_back(temp);
        }
        P.push_back(tempVec);
    }
}

void nn::connect() {
    for (uzi i = 0; i < model.size() - 1; i++) {
        for (uzi j = 0; j < model[i]; j++) {
            for (uzi k = 0; k < model[i + 1]; k++) {
                P[i][j]->connect(P[i + 1][k]->id);
            }
        }
    }
}

void nn::printInfo() {
    std::cout << "Learning MNIST Database\n" << std::endl;
    std::cout << "Network: {" << inputSize << ",";
    for (uzi i = 0; i < hiddenLayerSize - 2; i++) { std::cout << model[i + 1] << ","; }
    std::cout << outputSize << "}\n" << std::endl;

    // Listing configurations
    if (logic(0.25) == sigmoid(0.25) && logic(0.5) == sigmoid(0.5)) {
        std::cout << "Logic function: sigmoid" << std::endl;
    } else if (logic(0.25) == reLU(0.25) && logic(0.5) == reLU(0.5)) {
        std::cout << "Logic function: reLU" << std::endl;
    }
#ifdef BP_USE_PID
    std::cout << "Using PID for backpropagation. Kp: " << pid_P << ", Ki:" << pid_I << ", Kd: "
              << pid_D << std::endl;
#endif
#ifdef BP_BELLMAN_OPT
    std::cout<<"Using Bellman's optimization for backpropagation. Gamma: "<<gamma<<std::endl;
#endif
    std::cout << "Backpropagation shadow updating gain: " << zeta << std::endl;
#ifdef ADAPTIVE_TRAINING
    std::cout<<"Adaptive training is active! Learning rate (η) will be updated to minimize the RMS error. "<<std::endl;
#endif
    std::cout << std::endl;
}
