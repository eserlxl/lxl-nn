#include <nn.h>

void nn::create() {
    uzi id = 0;
    for (uzi i : model) {
        std::vector<neuron *> tempVec;
        for (uzi j = 0; j < i; j++) {
            auto *temp = new neuron(id++, 0.f);
            tempVec.push_back(temp);
        }
        P.push_back(tempVec);
    }
}

void nn::connect() {
    for (uzi i = 0; i < layerCount - 1; i++) {
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
    for (uzi i = 0; i < layerCount - 2; i++) { std::cout << model[i + 1] << ","; }
    std::cout << outputSize << "}\n" << std::endl;

    // Listing configurations
    std::cout << "Learning Rate (η): " << eta << std::endl;
    //std::cout << "Learning Momentum (α): " << alpha << std::endl;
#ifdef BP_USE_BIAS
    std::cout << "Backpropagation Bias Gain (ζ): " << zeta << std::endl;
#endif

#ifdef LOGIC_SIGMOID
    std::cout << "Logic function: sigmoid" << std::endl;
#elif defined(LOGIC_RELU)
    std::cout << "Logic function: reLU" << std::endl;
#endif

#ifdef BP_USE_PID
    std::cout << "Using PID for backpropagation. Kp: " << pid_P << ", Ki:" << pid_I << ", Kd: "
              << pid_D << std::endl;
#endif
#ifdef BP_BELLMAN_OPT
    std::cout << "Using Bellman's optimization for backpropagation. Bellman's gain (γ): " << gamma << std::endl;
#endif
#ifdef ADAPTIVE_LEARNING
    std::cout
            << "Adaptive training is active! Learning rate of weights (η) and learning rate of biases (ζ) will be updated to minimize the RMS error. "
            << "\nAdaptive Learning Gain (α): " << alpha << std::endl;
#endif
    std::cout << std::endl;
}
