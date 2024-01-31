#include "nn/neuralNetwork.h"

void NeuralNetwork::init() {

#ifdef NO_RANDOMIZATION
    seed = 1;
#else
    seed = rd();
#endif
    e2.seed(seed);
#ifdef BP_USE_BIAS
    bias.resize(outputIndex);

    for (uzi k = 0; k < outputIndex; k++) {
        bias[k].resize(model[k + 1]);
    }
#endif
    errorBP.resize(layerCount);
    prevError.resize(outputSize);

    learningMatrix.push_back(10.f); // Learning Rate of weights

    uzi learningRateIndex = 0;
    weightLearningRateIndex = learningRateIndex;
    learningRateIndex++;
#ifdef BP_USE_BIAS
    learningMatrix.push_back(8.75f); // Learning Rate of biases
    biasLearningRateIndex = learningRateIndex;
    learningRateIndex++;
#endif
#ifdef BP_BELLMAN_OPT
    learningMatrix.push_back(0.25f); // Bellman's optimality gain
    bellmanLearningRateIndex = learningRateIndex;
    learningRateIndex++;
#endif
#ifdef ADAPTIVE_LEARNING
    learningMatrix.push_back(0.05f); // Adaptive learning gain
    adaptiveLearningRateIndex = learningRateIndex;

    learningMatrix.push_back(0.05f); // Adaptive learning rate limit
    rateLimitLearningRateIndex = learningRateIndex;

    learningMatrixLowerLimits.push_back(
            learningMatrix[weightLearningRateIndex] * 0.95f); // min Learning Rate of weights
    learningMatrixUpperLimits.push_back(learningMatrix[weightLearningRateIndex] * 1.5f); // max Learning Rate of weights
#ifdef BP_USE_BIAS
    learningMatrixLowerLimits.push_back(learningMatrix[biasLearningRateIndex] * 0.5f); // min Learning Rate of biases
    learningMatrixUpperLimits.push_back(learningMatrix[biasLearningRateIndex] * 1.5f); // max Learning Rate of biases
#endif
#ifdef BP_BELLMAN_OPT
    learningMatrixLowerLimits.push_back(0.125f); // min Bellman's optimality gain
    learningMatrixUpperLimits.push_back(0.33f); // max Bellman's optimality gain
#endif
    learningMatrixLowerLimits.push_back(
            learningMatrix[adaptiveLearningRateIndex] * 0.75f); // min adaptive learning gain
    learningMatrixUpperLimits.push_back(learningMatrix[adaptiveLearningRateIndex] * 1.25f);// max adaptive learning gain

    learningMatrixLowerLimits.push_back(
            learningMatrix[rateLimitLearningRateIndex] * 0.75f); // min adaptive learning rate limit
    learningMatrixUpperLimits.push_back(
            learningMatrix[rateLimitLearningRateIndex] * 1.25f);// max adaptive learning rate limit
#endif

#ifdef BP_USE_PID
    errorSumBP.resize(outputSize);
        pid_P = -.5e-3f;
        pid_I = -0.125f / (float) sourceSize;
        pid_D = 0;//-0.01;//0.1;//0.5f;
#endif

#ifdef ADAPTIVE_LEARNING
    weightBackupCount = 0;
#endif
}

void NeuralNetwork::setIO(std::vector<float> input, std::vector<float> output) {
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
