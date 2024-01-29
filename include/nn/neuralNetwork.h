#ifndef lxl_nn_NeuralNetwork_H_
#define lxl_nn_NeuralNetwork_H_

#include <iostream>
#include <random>
#include <Eigen/Dense>
#include <lxl.h>

using namespace lxl;

#include <nn/neuron.h>
#include <dataLoader.h>
#include "test.h"

#define NO_RANDOMIZATION // Only for testing the algorithm, we need the same results for each run to compare.

//#define BP_BELLMAN_OPT
//#define BP_USE_BIAS
//#define ADAPTIVE_LEARNING
//#define BP_USE_PID

#define ANALYSE_TRAINING

// LOGIC_SIGMOID, LOGIC_TANH or LOGIC_RELU
#define LOGIC_SIGMOID [0,1]
//#define LOGIC_TANH // [-1,1]
//#define LOGIC_SWISH
//#define LOGIC_RELU

typedef std::size_t uzi;

class NeuralNetwork {
public:
    std::random_device rd;
    std::mt19937 e2;
    std::vector<std::vector<neuron *>> P;
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
        learningMatrix.push_back(0.125f); // Adaptive learning gain
        adaptiveLearningRateIndex = learningRateIndex;

        learningMatrix.push_back(0.05f); // Adaptive learning rate limit
        rateLimitLearningRateIndex = learningRateIndex;

        learningMatrixLowerLimits.push_back(10.f); // min Learning Rate of weights
        learningMatrixUpperLimits.push_back(15.f); // max Learning Rate of weights
#ifdef BP_USE_BIAS
        learningMatrixLowerLimits.push_back(2.5f); // min Learning Rate of biases
        learningMatrixUpperLimits.push_back(20.f); // max Learning Rate of biases
#endif
#ifdef BP_BELLMAN_OPT
        learningMatrixLowerLimits.push_back(0.125f); // min Bellman's optimality gain
        learningMatrixUpperLimits.push_back(0.33f); // max Bellman's optimality gain
#endif
        learningMatrixLowerLimits.push_back(0.125f); // min adaptive learning gain
        learningMatrixUpperLimits.push_back(0.25f);// max adaptive learning gain

        learningMatrixLowerLimits.push_back(0.025f); // min adaptive learning rate limit
        learningMatrixUpperLimits.push_back(0.075f);// max adaptive learning rate limit
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

    ~NeuralNetwork() {
        delete (clock);
        delete (chronometer);
    }

    void create();

    void connect();

    void randWeights();

    void feedForward();

    void backPropagateOutputLayer();

    void backPropagate();

    void setIO(std::vector<float> input, std::vector<float> output);

    void train(uzi loopMax, MNISTData *testData);

    TestResult checkTrainingData();

    TestResult checkTestData(MNISTData *testData);

    void initValues();

    void printInfo();

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

    uzi correctChoice;
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

    std::vector<std::vector<float>> inputStructure; // { min, max }
    std::vector<std::vector<float>> outputStructure; // { min, max }

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

    float convertSourceNorm(float x, uzi j);

    float convertTargetNorm(float x, uzi j);

    void normIO(std::vector<std::vector<float>> input, std::vector<std::vector<float>> output);
};

#endif // lxl_nn_NeuralNetwork_H_
