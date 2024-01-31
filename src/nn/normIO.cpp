#include "nn/neuralNetwork.h"

float NeuralNetwork::convertSourceToInput(float x) const {
    return (x - inputMinValue) / inputRange * networkRange + networkMinValue;
}

float NeuralNetwork::convertTargetToOutput(float x) const {
    return (x - networkMinValue) / networkRange * outputRange + outputMinValue;
}

float NeuralNetwork::convertTargetDiffToOutputDiff(float x) const {
    return x / networkRange * outputRange;
}

float NeuralNetwork::convertOutputToTarget(float x) const {
    return (x - outputMinValue) / outputRange * networkRange + networkMinValue;
}

void NeuralNetwork::setInput(std::vector<float> input) {
    if (input.size() != model[0]) {
        std::cout << "Improper network design according to inputs!" << std::endl;
        exit(-1);
    }
    for (uzi i = 0; i < model[0]; i++) {
        network[0][i] = convertSourceToInput(input[i]);
        P[0][i]->value = network[0][i];
    }
}

std::vector<float> NeuralNetwork::getOutput() {
    std::vector<float> tempVec;
    for (uzi i = 0; i < target[0].size(); i++) {
        tempVec.push_back(convertTargetToOutput(P[outputIndex][i]->value));
    }
    return tempVec;
}


void NeuralNetwork::normIO(std::vector<std::vector<float>> input, std::vector<std::vector<float>> output) {

    inputMinValue = lxl::min(input)[0];
    inputMaxValue = lxl::max(input)[0];
    inputRange = inputMaxValue - inputMinValue;
    outputMinValue = lxl::min(output)[0];
    outputMaxValue = lxl::max(output)[0];
    outputRange = outputMaxValue - outputMinValue;

    sourceSize = input.size();
    targetSize = output.size();
    source.resize(sourceSize);
    target.resize(targetSize);

    for (uzi i = 0; i < sourceSize; i++) {
        source[i].resize(input[i].size());
        for (uzi j = 0; j < input[i].size(); j++) {
            source[i][j] = networkRange * (input[i][j] - inputMinValue) / inputRange + networkMinValue;
        }
    }
    for (uzi i = 0; i < targetSize; i++) {
        target[i].resize(output[i].size());
        for (uzi j = 0; j < output[i].size(); j++) {
            target[i][j] = networkRange * (output[i][j] - outputMinValue) / outputRange + networkMinValue;
        }
    }
}