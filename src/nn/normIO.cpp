#include "nn/neuralNetwork.h"

float NeuralNetwork::convertSourceToInput(float x, uzi j) const {
    return (x - networkMinValue) / networkRange * (inputMaxValue - inputMinValue) +
            inputMinValue;
}

float NeuralNetwork::convertTargetToOutput(float x, uzi j) const {
    return (x - networkMinValue) / networkRange *
           (outputMaxValue - outputMinValue) + outputMinValue;
}

void NeuralNetwork::normIO(std::vector<std::vector<float>> input, std::vector<std::vector<float>> output) {

    inputMinValue = lxl::min(input)[0];
    inputMaxValue = lxl::max(input)[0];
    outputMinValue = lxl::min(output)[0];
    outputMaxValue = lxl::max(output)[0];

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