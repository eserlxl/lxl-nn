#include "nn/neuralNetwork.h"

void NeuralNetwork::feedForward() {
    for (uzi i = 0; i < outputIndex; i++) {
        for (uzi j = 0; j < model[i + 1]; j++) {
#ifdef BP_USE_BIAS
            float tempSum = bias[i][j];
#else
            float tempSum = 0;
#endif
            for (uzi k = 0; k < model[i]; k++) {
                tempSum += P[i][k]->value * P[i][k]->Link[j]->weight;
            }
            float x = tempSum / (float) (model[i] + 1);
#ifdef LOGIC_SIGMOID
            P[i + 1][j]->value = sigmoid(x);
#elif defined(LOGIC_TANH)
            P[i + 1][j]->value = std::tanh(x);
#elif defined(LOGIC_SWISH)
            P[i + 1][j]->value = std::max(0.f,std::min(x*sigmoid(x),1.f));
#elif defined(LOGIC_RELU)
            P[i + 1][j]->value = reLU(x);
#endif
        }
    }
}

float NeuralNetwork::calcRMSE() {
    matrixFloat1D tempVec;
    for (uzi s = 0; s < sourceSize; s++) {

        network[0] = source[s];
        for (uzi i = 0; i < model[0]; i++) {
            P[0][i]->value = network[0][i];
        }
        feedForward();

        matrixFloat1D tempVec2;
        for (uzi j = 0; j < model[outputIndex]; j++) {
            tempVec2.push_back(convertTargetDiffToOutputDiff(P[outputIndex][j]->value - target[s][j]));
        }
        tempVec.push_back(rms(tempVec2));
        networkOutput.push_back(getOutput());
    }
    return rms(tempVec);
}

float NeuralNetwork::calcNormRMSE() {
    return calcRMSE() / outputMaxValue;
}

float NeuralNetwork::calcNormRMSEPercentage() {
    return calcNormRMSE() * 100.f;
}