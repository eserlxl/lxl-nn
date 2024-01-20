#ifndef lxl_nn_NN_H_
#define lxl_nn_NN_H_

#include <iostream>
#include <random>
#include <nn/perceptron.h>
#include <dataLoader.h>
#include "timer.h"
#include "test.h"

#define NO_RANDOMIZATION // Only for testing the algorithm, we need the same results for each run to compare.

#define BP_BELLMAN_OPT
//#define BP_USE_PID
//#define ADAPTIVE_LEARNING

#define ANALYSE_TRAINING

/**
 * Sigmoid
 * Simple logistic function, It is a smooth, S-shaped curve.
 * Output: [0,1]
 */
template<typename Float>
Float sigmoid(Float x, Float a = 1) {
    return (Float) 1 / ((Float) 1 + std::exp(-a * x));
}

// Rectified Linear Unit
// x<0?0:(x>1?1:x)
template<typename Float>
Float reLU(Float x) {
    return std::min(std::max((Float) 0., x), (Float) 1);
}

#define logic sigmoid

template<typename T>
T rms(std::vector<T> &array) {
    T temp = 0.;

    int n = array.size();

    for (int i = 0; i < n; i++) {
        temp += std::pow(array[i], (T) 2);
    }

    return std::sqrt(temp / (T) n);
}

typedef std::size_t uzi;

class nn {
public:
    std::random_device rd;
    std::mt19937 e2;
    std::vector<std::vector<perceptron *>> P;
    uzi outputIndex;
    float eta;
    uzi sourceSize;
    uzi targetSize;
    uzi hiddenLayerSize;
    uzi inputSize;
    uzi outputSize;
    std::vector<uzi> model;

    nn(std::vector<uzi> model, const std::vector<std::vector<float>> &input,
       const std::vector<std::vector<float>> &output) {
        clock = new Timer();
        chronometer = new Timer();

        source = input;
        target = output;
        sourceSize = source.size();
        targetSize = target.size();

        this->model = model;
        network.resize(model.size());
        uzi layerIndex = 0;
        for (unsigned long layer : model) {
            network[layerIndex++].resize(layer);
        }

        layerCount = network.size();
        hiddenLayerSize = layerCount - 2;
        outputIndex = layerCount - 1;
        inputSize = model[0];
        outputSize = model[outputIndex];

        create();
        connect();

#ifdef NO_RANDOMIZATION
        e2.seed(1);
#else
        e2.seed(rd());
#endif
        shadow.resize(outputIndex);
        deltaShadow.resize(outputIndex);

        for (uzi k = 0; k < outputIndex; k++) {
            shadow[k].resize(model[k]);
            deltaShadow[k].resize(model[k]);
        }

        errorBP.resize(network.size());
        prevError.resize(outputSize);

#ifdef BP_BELLMAN_OPT
        gamma = 0.25; // 0 < gamma <= 1
#endif
        zeta = 0.875;
#ifdef BP_USE_PID
        errorSumBP.resize(outputSize);
        pid_P = -.5e-3f;
        pid_I = -0.125f / (float) sourceSize;
        pid_D = 0;//-0.01;//0.1;//0.5f;
#endif
        initEtaAlpha();

#ifdef ADAPTIVE_LEARNING
        weightBackupCount=0;
#endif
    }

    ~nn() {
        delete (clock);
        delete (chronometer);
    }

    void create();

    void connect();

    void randWeights();

    void feedForward();

    void backPropagateInit();

    void backPropagate();

    void setIO(std::vector<float> input, std::vector<float> output);

    void train(uzi loopMax, MNISTData *testData);

    TestResult checkTrainingData();

    TestResult checkTestData(MNISTData *testData);

    void initValues();

    void initEtaAlpha();

    void printInfo();

private:
    uzi layerCount;
    std::vector<std::vector<float>> source;
    std::vector<std::vector<float>> target;
    std::vector<std::vector<float>> network;
    std::vector<std::vector<float>> shadow;
    std::vector<std::vector<float>> deltaShadow;
#ifdef BP_USE_PID
    std::vector<float> errorSumBP;
    float pid_P;
    float pid_I;
    float pid_D;
#endif
#ifdef BP_BELLMAN_OPT
    float gamma; // 0 < gamma <= 1
#endif
    float zeta;
    Timer *clock;
    Timer *chronometer;
    float rmsErrorBP;
    std::vector<std::vector<float>> errorBP;
    std::vector<float> prevError;

    uzi correctChoice;
#ifdef ADAPTIVE_LEARNING
    std::vector<std::vector<float>> pBackup;
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
    void predictWeights();
#endif
};

#endif // lxl_nn_NN_H_
