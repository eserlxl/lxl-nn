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


//#define NO_RANDOMIZATION // Only for testing the algorithm, we need the same results for each run to compare.

#define BINARY_OUTPUT_DATA // For networks that have 0 or 1 outputs only

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

    NeuralNetwork(const matrixUzi1D &model, const matrixFloat2D &input = {{}}, const matrixFloat2D &output = {{}}) {
        clock = new lxl::Timer();
        chronometer = new lxl::Timer();

        this->model = model;

        preInit();

        create();
        connect();
        init();

        if(!input.empty() && !output.empty()){
            normIO(input, output);
        }
    }

    template <typename T, typename = std::string> NeuralNetwork(const matrixUzi1D &model, const T &fileName) {
        clock = new lxl::Timer();
        chronometer = new lxl::Timer();

        this->model = model;

        preInit();

        create();
        connect();
        init();

        loadDataFromFile(fileName);
    }

    template <typename T, typename = std::string> NeuralNetwork(const T &fileName) {
        clock = new lxl::Timer();
        chronometer = new lxl::Timer();

        matrixFloat2D data;

        std::string delimiter;
        lxl::detectDelimiter(fileName, &delimiter);
        lxl::fetchData(fileName, data, delimiter);

        load(fileName);

        libFile = fileName;

        preInit();

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
        saveWeights();
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

    void train(uzi loopMax);

    float checkBinaryOutputData();

    void printNetworkInfo();

    void normIO(matrixFloat2D input, matrixFloat2D output);

    float calcRMSE();

    float calcNormRMSE();

    float calcNormRMSEPercentage();

    void loadDataFromFile(const std::string &fileName);

    float NRMSEPercentage;
#ifdef BINARY_OUTPUT_DATA
    float binaryDataErrorPercentage;
#endif

private:
    uzi layerCount;
    uzi maxLayerSize;
    matrixFloat2D source;
    matrixFloat2D target;
    matrixFloat2D network;
    matrixFloat2D networkOutput;
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

    matrixFloat1D learningMatrixUpperLimits;
    matrixFloat1D learningMatrixLowerLimits;

    matrixFloat1D weightBackup = {};

    void saveWeights();

    void loadWeights();

    void smoothWeights(float backupRatio);

    std::string libFile;

#endif

    float randomNumber();

    float randomNumberExtended();

    void preInit();

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

    float convertInputToSource(float x) const;

    float convertSourceToInput(float x) const;

    float convertTargetToOutput(float x) const;

    float convertOutputToTarget(float x) const;

    float convertTargetDiffToOutputDiff(float x) const;

    void setLearningMatrix();

    void setLearningMatrixLimits();
};

#endif // lxl_nn_NeuralNetwork_H_
