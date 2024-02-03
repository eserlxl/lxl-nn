#include "nn/neuralNetwork.h"

void NeuralNetwork::create() {
    uzi id = 0;
    for (uzi i : model) {
        matrixNeuron1D tempVec;
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

/**
 * Saves the neural network to the given text file.
 * @param fileName
 */
void NeuralNetwork::save(const std::string &fileName) {
    std::filesystem::path path(fileName);
    if (!std::filesystem::exists(path.parent_path())) {
        std::filesystem::create_directory(path.parent_path());
    }

    std::stringstream fP;

    uzi precision = 8;

    matrixFloat1D errorVec;
    errorVec.push_back(NRMSEPercentage);
#ifdef BINARY_OUTPUT_DATA
    errorVec.push_back(binaryDataErrorPercentage);
#endif
    print(model, "model", 0, fP);
    print(learningMatrix, "learningMatrix", precision, fP);

    matrixFloat1D tempVec = {
            inputMinValue,
            inputMaxValue,
            inputRange,
            outputMinValue,
            outputMaxValue,
            outputRange
    };
    print(tempVec, "iOStructure", precision, fP);

    print(errorVec, "error", precision, fP);

    matrixFloat1D valueVec;

#ifdef BP_USE_BIAS
    for (uzi i = 0; i < outputIndex; i++) {
        for (uzi j = 0; j < P[i + 1].size(); j++) {
            valueVec.push_back(bias[i][j]);
        }
    }
#endif

    for (uzi i = 0; i < model.size(); i++) {
        for (uzi j = 0; j < model[i]; j++) {
            valueVec.push_back(P[i][j]->value);
            if (i < outputIndex) {
                for (uzi k = 0; k < model[i + 1]; k++) {
                    valueVec.push_back(P[i][j]->Link[k]->weight);
                }
            }
        }
    }
    print(valueVec, "network", precision, fP);

    Gzip::compress(fP, fileName);
}

void NeuralNetwork::load(const std::string &fileName) {
    matrixString2D data;
    fetchDataGzip(fileName, data, " ");

    for (uzi d = 0; d < data.size(); d++) {

        uzi dim = convert<uzi>(data[d][0]);
        matrixUzi1D sizeVec;

        for (uzi j = 0; j < dim; j++) {
            sizeVec.push_back(convert<uzi>(data[d][1 + j]));
        }

        // Skip empty data
        if (sizeVec.empty() || sizeVec[0] == 0) {
            continue;
        }

        std::string text = data[d][1 + dim];

        if (text == "model") {
            model.clear();
            for (auto &x : data[d + 1]) {
                model.push_back(convert<uzi>(x));
            }
            preInit();
        } else if (text == "learningMatrix") {
            setLearningMatrix(); // For getting indexes only
            learningMatrix.clear();
            for (auto &x : data[d + 1]) {
                learningMatrix.push_back(convert<float>(x));
            }
        } else if (text == "iOStructure") {
            matrixFloat1D iOStructure;
            for (auto &x : data[d + 1]) {
                iOStructure.push_back(convert<float>(x));
            }

            uzi h = 0;
            inputMinValue = iOStructure[h++];
            inputMaxValue = iOStructure[h++];
            inputRange = iOStructure[h++];
            outputMinValue = iOStructure[h++];
            outputMaxValue = iOStructure[h++];
            outputRange = iOStructure[h++];
        } else if (text == "network") {
            matrixFloat1D networkData;
            for (auto &x : data[d + 1]) {
                networkData.push_back(convert<float>(x));
            }

            P.clear();

            uzi h = 0;
            outputIndex = model.size() - 1;
#ifdef BP_USE_BIAS
            bias.resize(outputIndex);
            for (uzi i = 0; i < outputIndex; i++) {
                bias[i].clear();
                for (uzi j = 0; j < model[i + 1]; j++) {
                    bias[i].push_back(networkData[h++]);
                }
            }
#endif
            for (uzi i = 0; i < model.size(); i++) {
                matrixNeuron1D tempLayer;
                for (uzi j = 0; j < model[i]; j++) {
                    uzi id = i * model[i] + j;
                    auto *tempP = new Neuron(id, networkData[h++]);
                    if (i < outputIndex) {
                        for (uzi k = 0; k < model[i + 1]; k++) {
                            auto *tempLink = new Synapse(id, id + 1 + k);
                            tempLink->weight = networkData[h++];
                            tempP->Link.push_back(tempLink);
                        }
                    }
                    tempLayer.push_back(tempP);
                }
                P.push_back(tempLayer);
            }
        }

        if (dim == 1) {
            d++;
        } else {
            d += sizeVec[0];
        }
    }
}

void NeuralNetwork::loadDataFromFile(const std::string &fileName) {
    matrixFloat2D data, input, output;

    std::string delimiter;
    lxl::detectDelimiter(fileName, &delimiter);
    lxl::fetchData(fileName, data, delimiter);

    if (outputSize != data[0].size() - inputSize) {
        std::cout << "Invalid data format! Input Size: " << inputSize << ", Output Size: " << outputSize
                  << ", Data size in a row: " << data[0].size() << std::endl;
        exit(-1);
    }

    for (auto &i : data) {
        matrixFloat1D temp;
        for (uzi j = 0; j < inputSize; j++) {
            temp.push_back(i[j]);
        }
        input.push_back(temp);

        temp.clear();
        for (uzi j = inputSize; j < inputSize + outputSize; j++) {
            temp.push_back(i[j]);
        }
        output.push_back(temp);
    }

    normIO(input, output);
}

void NeuralNetwork::printNetworkInfo() {
#ifdef BINARY_OUTPUT_DATA
    std::cout << "Learning MNIST Database\n" << std::endl;
#endif
    std::cout << "Network: {" << inputSize << ",";
    for (uzi i = 0; i < layerCount - 2; i++) { std::cout << model[i + 1] << ","; }
    std::cout << outputSize << "}\n" << std::endl;

    if (!libFile.empty()) {
        std::cout << "Network library file: " << libFile << "\n" << std::endl;
    }

    // Listing configurations
#ifdef BINARY_OUTPUT_DATA
    std::cout << "Binary output data!\n" << std::endl;
#endif
#ifdef NO_RANDOMIZATION
    std::cout << "Randomization is disabled!. Seed: " << seed << std::endl;
#else
    std::cout << "Randomization is enabled!. Seed: "<<seed<< std::endl;
#endif
#ifdef ADAPTIVE_LEARNING
    std::cout
            << "\nAdaptive learning is active!\nLearning rates will be updated to minimize the RMS error according to given ranges.\n"
            << "\nLearning rate of weights (η): " << learningMatrix[weightLearningRateIndex] << ", range: " << "["
            << learningMatrixLowerLimits[weightLearningRateIndex] << ", "
            << learningMatrixUpperLimits[weightLearningRateIndex] << "]"
            #ifdef BP_USE_BIAS
            << "\nLearning rate of biases (ζ): " << learningMatrix[biasLearningRateIndex] << ", range: " << "["
            << learningMatrixLowerLimits[biasLearningRateIndex] << ", "
            << learningMatrixUpperLimits[biasLearningRateIndex] << "]"
            #endif
            #ifdef BP_BELLMAN_OPT
            << "\nBellman's optimality gain (γ0): " << learningMatrix[bellman0LearningRateIndex] << ", range: " << "["
            << learningMatrixLowerLimits[bellman0LearningRateIndex] << ", "
            << learningMatrixUpperLimits[bellman0LearningRateIndex] << "]"

            << "\nBellman's optimality gain (γ1): " << learningMatrix[bellman1LearningRateIndex] << ", range: " << "["
            << learningMatrixLowerLimits[bellman1LearningRateIndex] << ", "
            << learningMatrixUpperLimits[bellman1LearningRateIndex] << "]"

            << "\nBellman's optimality gain (γ2): " << learningMatrix[bellman2LearningRateIndex] << ", range: " << "["
            << learningMatrixLowerLimits[bellman2LearningRateIndex] << ", "
            << learningMatrixUpperLimits[bellman2LearningRateIndex] << "]"
            #endif
            << "\nAdaptive learning gain (α): " << learningMatrix[adaptiveLearningRateIndex] << ", range: " << "["
            << learningMatrixLowerLimits[adaptiveLearningRateIndex] << ", "
            << learningMatrixUpperLimits[adaptiveLearningRateIndex] << "]"

            << "\nAdaptive learning Smooth Weight Threshold (SWt): " << learningMatrix[adaptiveLearningSWThresholdIndex]
            << ", range: " << "["
            << learningMatrixLowerLimits[adaptiveLearningSWThresholdIndex] << ", "
            << learningMatrixUpperLimits[adaptiveLearningSWThresholdIndex] << "]"

            << "\nAdaptive learning Smooth Weight Backup Ratio (SWr): "
            << learningMatrix[adaptiveLearningSWBackupRatioIndex] << ", range: " << "["
            << learningMatrixLowerLimits[adaptiveLearningSWBackupRatioIndex] << ", "
            << learningMatrixUpperLimits[adaptiveLearningSWBackupRatioIndex] << "]"

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

    std::cout << "inputMinValue: " << inputMinValue << std::endl;
    std::cout << "inputMaxValue: " << inputMaxValue << std::endl;
    std::cout << "inputRange: " << inputRange << std::endl;
    std::cout << "outputMinValue: " << outputMinValue << std::endl;
    std::cout << "outputMaxValue: " << outputMaxValue << std::endl;
    std::cout << "outputRange: " << outputRange << std::endl;

    std::cout << std::endl;
}