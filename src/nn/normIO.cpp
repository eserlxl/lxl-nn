#include "nn/neuralNetwork.h"

float NeuralNetwork::convertSourceNorm(float x, uzi j) {
    return (x - networkMinValue) / (networkMaxValue - networkMinValue) * (inputStructure[j][1] - inputStructure[j][0]) +
           inputStructure[j][0];
}

float NeuralNetwork::convertTargetNorm(float x, uzi j) {
    return (x - networkMinValue) / (networkMaxValue - networkMinValue) *
           (outputStructure[j][1] - outputStructure[j][0]) + outputStructure[j][0];
}

void NeuralNetwork::normIO(std::vector<std::vector<float>> input, std::vector<std::vector<float>> output) {

    float inputMinValue = lxl::min(input)[0];
    float inputMaxValue = lxl::max(input)[0];
    float outputMinValue = lxl::min(output)[0];
    float outputMaxValue = lxl::max(output)[0];

    sourceSize = input.size();
    targetSize = output.size();
    source.resize(sourceSize);
    target.resize(targetSize);

    for (uzi i = 0; i < sourceSize; i++) {
        source[i].resize(input[i].size());
        for (uzi j = 0; j < input[i].size(); j++) {
            source[i][j] = networkRange * (input[i][j] - inputMinValue) / (inputMaxValue - inputMinValue) +
                           networkMinValue;
        }
    }
    for (uzi i = 0; i < targetSize; i++) {
        target[i].resize(output[i].size());
        for (uzi j = 0; j < output[i].size(); j++) {
            target[i][j] = networkRange * (output[i][j] - outputMinValue) / (outputMaxValue - outputMinValue) +
                           networkMinValue;
        }
    }
}