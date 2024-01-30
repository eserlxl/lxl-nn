#include "nn/neuralNetwork.h"

void NeuralNetwork::create() {
    uzi id = 0;
    for (uzi i : model) {
        std::vector<Neuron *> tempVec;
        for (uzi j = 0; j < i; j++) {
            auto *temp = new Neuron(id++, 0.f);
            tempVec.push_back(temp);
        }
        P.push_back(tempVec);
    }
}

void NeuralNetwork::connect() {
    for (uzi i = 0; i < layerCount - 1; i++) {
        for (uzi j = 0; j < model[i]; j++) {
            for (uzi k = 0; k < model[i + 1]; k++) {
                P[i][j]->connect(P[i + 1][k]->id);
            }
        }
    }
}

float NeuralNetwork::randomNumberExtended() {
    float x = randomNumber();
    while (std::fabs(x - 0.5f) < 1e-3) {
        x = randomNumber();
    }
    return 2.f * (x - 0.5f);
}

float NeuralNetwork::randomNumber() {
    float x = e2() / (float) std::mt19937::max();
    while (x < 1e-3) {
        x = randomNumber();
    }
    return x;
}

void NeuralNetwork::printNetworkInfo() {
    std::cout << "Learning MNIST Database\n" << std::endl;
    std::cout << "Network: {" << inputSize << ",";
    for (uzi i = 0; i < layerCount - 2; i++) { std::cout << model[i + 1] << ","; }
    std::cout << outputSize << "}\n" << std::endl;

    // Listing configurations
#ifdef NO_RANDOMIZATION
    std::cout << "Randomization is disabled!. Seed: " << seed << std::endl;
#else
    std::cout << "Randomization is enabled!. Seed: "<<seed<< std::endl;
#endif
#ifdef ADAPTIVE_LEARNING
    std::cout
            << "Adaptive learning is active!\nLearning rates will be updated to minimize the RMS error according to given ranges.\n"
            << "\nLearning rate of weights (η): " << learningMatrix[weightLearningRateIndex] << ", range: " << "["
            << learningMatrixLowerLimits[weightLearningRateIndex] << ", "
            << learningMatrixUpperLimits[weightLearningRateIndex] << "]"
            #ifdef BP_USE_BIAS
            << "\nLearning rate of biases (ζ): " << learningMatrix[biasLearningRateIndex] << ", range: " << "["
            << learningMatrixLowerLimits[biasLearningRateIndex] << ", "
            << learningMatrixUpperLimits[biasLearningRateIndex] << "]"
            #endif
            #ifdef BP_BELLMAN_OPT
            << "\nBellman's optimality gain (γ): " << learningMatrix[bellmanLearningRateIndex] << ", range: " << "["
            << learningMatrixLowerLimits[bellmanLearningRateIndex] << ", "
            << learningMatrixUpperLimits[bellmanLearningRateIndex] << "]"
            #endif
            << "\nAdaptive learning gain (α): " << learningMatrix[adaptiveLearningRateIndex] << ", range: " << "["
            << learningMatrixLowerLimits[adaptiveLearningRateIndex] << ", "
            << learningMatrixUpperLimits[adaptiveLearningRateIndex] << "]"
            << "\nRate limit (r): " << learningMatrix[rateLimitLearningRateIndex] << ", range: " << "["
            << learningMatrixLowerLimits[rateLimitLearningRateIndex] << ", "
            << learningMatrixUpperLimits[rateLimitLearningRateIndex] << "]"
            << "\n" << std::endl;
#else
    std::cout << "Learning rate of weights (η): " << learningMatrix[weightLearningRateIndex] << std::endl;
#ifdef BP_USE_BIAS
    std::cout << "Learning rate of biases (ζ): " << learningMatrix[biasLearningRateIndex] << std::endl;
#endif
#ifdef BP_BELLMAN_OPT
    std::cout << "Using Bellman's optimization for backpropagation. Bellman's gain (γ): " << learningMatrix[bellmanLearningRateIndex]
              << std::endl;
#endif
#endif


#ifdef LOGIC_SIGMOID
    std::cout << "Logic function: sigmoid" << std::endl;
#elif defined(LOGIC_TANH)
    std::cout << "Logic function: Tanh" << std::endl;
#elif defined(LOGIC_SWISH)
    std::cout << "Logic function: Swish" << std::endl;
#elif defined(LOGIC_RELU)
    std::cout << "Logic function: reLU" << std::endl;
#endif

#ifdef BP_USE_PID
    std::cout << "Using PID for backpropagation. Kp: " << pid_P << ", Ki:" << pid_I << ", Kd: "
              << pid_D << std::endl;
#endif

    std::cout << std::endl;
}
