#ifndef lxl_nn_NeuralNetwork_H_
#define lxl_nn_NeuralNetwork_H_

#include <iostream>
#include <random>
#include <Eigen/Dense>
#include <lxl.h>

using namespace lxl;

#include "neuron.h"
#include "test/test.h"
#include "test/analyse.h"


#define NO_RANDOMIZATION // Only for testing the algorithm, we need the same results for each run to compare.

//#define LEARNING_MNIST_DATA

//#define LOGIC_NETWORK

#define BP_BELLMAN_OPT
#define BP_USE_BIAS
#define ADAPTIVE_LEARNING
//#define BP_USE_PID

#define ANALYSE_TRAINING

// LOGIC_SIGMOID, LOGIC_TANH or LOGIC_RELU
#define LOGIC_SIGMOID [0,1]
//#define LOGIC_TANH // [-1,1]
//#define LOGIC_SWISH
//#define LOGIC_RELU

class NeuralNetwork {
public:
    std::random_device rd;
    std::mt19937 e2;
    std::vector<std::vector<Neuron *>> P;
    uzi outputIndex;
    uzi sourceSize;
    uzi targetSize;
    uzi hiddenLayerSize;
    uzi inputSize;
    uzi outputSize;
    std::vector<uzi> model;

    NeuralNetwork(std::vector<uzi> model, const std::vector<std::vector<float>> &input,
                  const std::vector<std::vector<float>> &output) {
        clock = new lxl::Timer();
        chronometer = new lxl::Timer();

        this->model = model;
        network.resize(model.size());
        uzi layerIndex = 0;
        maxLayerSize = 0;
        for (uzi layer : model) {
            if (maxLayerSize < layer) {
                maxLayerSize = layer;
            }
            network[layerIndex++].resize(layer);
        }

        layerCount = network.size();
        hiddenLayerSize = layerCount - 2;
        outputIndex = layerCount - 1;
        inputSize = model[0];
        outputSize = model[outputIndex];

        normIO(input, output);
        create();
        connect();
        init();
    }

    NeuralNetwork(std::vector<uzi> model, const std::string &fileName) {
        clock = new lxl::Timer();
        chronometer = new lxl::Timer();

        std::vector<std::vector<float>> data, input, output;

        std::string delimiter;
        lxl::detectDelimiter(fileName, &delimiter);
        lxl::fetchData(fileName, data, delimiter);

        this->model = model;
        network.resize(model.size());
        uzi layerIndex = 0;
        maxLayerSize = 0;
        for (uzi layer : model) {
            if (maxLayerSize < layer) {
                maxLayerSize = layer;
            }
            network[layerIndex++].resize(layer);
        }

        layerCount = network.size();
        hiddenLayerSize = layerCount - 2;
        outputIndex = layerCount - 1;
        inputSize = model[0];
        outputSize = model[outputIndex];

        if (outputSize != data[0].size() - inputSize) {
            std::cout << "Invalid data format! Input Size: " << inputSize << ", Output Size: " << outputSize
                      << ", Data size in a row: " << data[0].size() << std::endl;
            exit(-1);
        }

        for (auto &i : data) {
            std::vector<float> temp;
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

        /*auto *analyse = new Analyse();
        analyse->detect(input, output);
        safeDelete(analyse);*/

        normIO(input, output);

        create();
        connect();
        init();
    }

    ~NeuralNetwork() {
        delete (clock);
        delete (chronometer);
    }

    void create();

    void connect();

    void init();

    void randWeights();

    void feedForward();

    void backPropagateOutputLayer();

    void backPropagate();

    void setIO(std::vector<float> input, std::vector<float> output);

#ifdef LEARNING_MNIST_DATA
    void train(uzi loopMax, MNISTData *testData);
#else

    void train(uzi loopMax);

#endif

    TestResult checkTrainingData();

    TestResult checkTestData(MNISTData *testData);

    void printNetworkInfo();

private:
    uzi layerCount;
    uzi maxLayerSize;
    std::vector<std::vector<float>> source;
    std::vector<std::vector<float>> target;
    std::vector<std::vector<float>> network;
    std::vector<std::vector<float>> bias;
    std::vector<float> learningMatrix;
#ifdef BP_USE_PID
    std::vector<float> errorSumBP;
    float pid_P;
    float pid_I;
    float pid_D;
#endif
    lxl::Timer *clock;
    lxl::Timer *chronometer;
    float rmsErrorBP;
    std::vector<std::vector<float>> errorBP;
    std::vector<float> prevError;

    float correctChoice;
    uzi weightLearningRateIndex;
#ifdef BP_USE_BIAS
    uzi biasLearningRateIndex;
#endif
#ifdef BP_BELLMAN_OPT
    uzi bellmanLearningRateIndex;
#endif
#ifdef ADAPTIVE_LEARNING
    uzi adaptiveLearningRateIndex;
    uzi rateLimitLearningRateIndex;

    std::vector<std::vector<float>> pBackup;

    std::vector<float> learningMatrixUpperLimits;
    std::vector<float> learningMatrixLowerLimits;

    void saveNetwork();

    void loadNetwork();

    std::vector<std::vector<float>> weightBackup;
    std::vector<float> lastWeightBackup;
    uzi weightBackupCount;

    void saveWeights();

    void saveWeights(uzi maxBackupCount);

    void loadWeights();

    void loadWeights(uzi backupIndex);

    void smoothWeights(float backupRatio);

    void smoothLastWeights();

#endif

    float randomNumber();

    float randomNumberExtended();

    uzi seed;

    float inputMinValue;
    float inputMaxValue;
    float inputRange;
    float outputMinValue;
    float outputMaxValue;
    float outputRange;

#ifdef LOGIC_SIGMOID
    float networkMinValue = 0;
    float networkMaxValue = 1;
#elif defined(LOGIC_TANH)
    float networkMinValue = -1;
    float networkMaxValue = 1;
#elif defined(LOGIC_SWISH)
    float networkMinValue = 0;
    float networkMaxValue = 1;
#elif defined(LOGIC_RELU)
    float networkMinValue = -1;
    float networkMaxValue = 1;
#endif
    float networkRange = networkMaxValue - networkMinValue;
    float networkMidRange = 0.5f * networkRange;
    float networkMidValue = networkMidRange + networkMinValue;
    float network3rdQuarterValue = 0.75f * networkRange + networkMinValue;
    float network1stQuarterValue = 0.25f * networkRange + networkMinValue;

    float convertSourceToInput(float x) const;

    float convertTargetToOutput(float x) const;

    float convertOutputToTarget(float x) const;

    float convertTargetDiffToOutputDiff(float x) const;

    void normIO(std::vector<std::vector<float>> input, std::vector<std::vector<float>> output);
};

#endif // lxl_nn_NeuralNetwork_H_
