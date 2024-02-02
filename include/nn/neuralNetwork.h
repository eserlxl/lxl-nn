#ifndef lxl_nn_NeuralNetwork_H_
#define lxl_nn_NeuralNetwork_H_

#include <iostream>
#include <random>
#include <filesystem>
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

//#define ANALYSE_TRAINING

// LOGIC_SIGMOID, LOGIC_TANH or LOGIC_RELU
#define LOGIC_SIGMOID [0,1]
//#define LOGIC_TANH // [-1,1]
//#define LOGIC_SWISH
//#define LOGIC_RELU

class NeuralNetwork {
public:
    typedef std::vector<Neuron *> matrixNeuron1D;
    typedef std::vector<matrixNeuron1D> matrixNeuron2D;

    std::random_device rd;
    std::mt19937 e2;
    matrixNeuron2D P;
    uzi outputIndex;
    uzi sourceSize;
    uzi targetSize;
    uzi hiddenLayerSize;
    uzi inputSize;
    uzi outputSize;
    matrixUzi1D model;

    NeuralNetwork(matrixUzi1D model, const matrixFloat2D &input,
                  const matrixFloat2D &output) {
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

    NeuralNetwork(matrixUzi1D model, const std::string &fileName) {
        clock = new lxl::Timer();
        chronometer = new lxl::Timer();

        matrixFloat2D data, input, output;

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

        /*auto *analyse = new Analyse();
        analyse->detect(input, output);
        safeDelete(analyse);*/

        normIO(input, output);

        create();
        connect();
        init();
    }

    NeuralNetwork(const std::string &fileName) {
        clock = new lxl::Timer();
        chronometer = new lxl::Timer();

        matrixFloat2D data, input, output;

        std::string delimiter;
        lxl::detectDelimiter(fileName, &delimiter);
        lxl::fetchData(fileName, data, delimiter);

        load(fileName);

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

        reqNormRMSE = {2.5e-3, 5e-3, 7.5e-3, 1e-2};

#ifdef NO_RANDOMIZATION
        seed = 1;
#else
        seed = rd();
#endif
        e2.seed(seed);

        errorBP.resize(layerCount);
        prevError.resize(outputSize);

        setLearningMatrixLimits();

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

    void save(const std::string &fileName);

    void load(const std::string &fileName);

    void create();

    void connect();

    void init();

    void randWeights();

    void feedForward();

    void backPropagateOutputLayer();

    void backPropagate();

    void setIO(const matrixFloat1D &input, const matrixFloat1D& output);

    void setInput(matrixFloat1D input);

    matrixFloat1D getOutput();

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
    matrixFloat2D source;
    matrixFloat2D target;
    matrixFloat2D network;
    matrixFloat2D bias;
    matrixFloat1D learningMatrix;
#ifdef BP_USE_PID
    matrixFloat1D errorSumBP;
    float pid_P;
    float pid_I;
    float pid_D;
#endif
    lxl::Timer *clock;
    lxl::Timer *chronometer;
    float rmsErrorBP;
    matrixFloat2D errorBP;
    matrixFloat1D prevError;

    matrixFloat1D reqNormRMSE;

    matrixFloat1D correctChoice;
    uzi weightLearningRateIndex;
#ifdef BP_USE_BIAS
    uzi biasLearningRateIndex;
#endif
#ifdef BP_BELLMAN_OPT
    uzi bellman0LearningRateIndex;
    uzi bellman1LearningRateIndex;
    uzi bellman2LearningRateIndex;
#endif
#ifdef ADAPTIVE_LEARNING
    uzi adaptiveLearningRateIndex;
    uzi adaptiveLearningSWThresholdIndex;
    uzi adaptiveLearningSWBackupRatioIndex;
    uzi rateLimitLearningRateIndex;

    matrixFloat2D pBackup;

    matrixFloat1D learningMatrixUpperLimits;
    matrixFloat1D learningMatrixLowerLimits;

    void saveNetwork();

    void loadNetwork();

    matrixFloat2D weightBackup;
    matrixFloat1D lastWeightBackup;
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

    void normIO(matrixFloat2D input, matrixFloat2D output);

    void setLearningMatrix();

    void setLearningMatrixLimits();
};

#endif // lxl_nn_NeuralNetwork_H_
