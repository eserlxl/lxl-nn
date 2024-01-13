#ifndef lxl_nn_NN_H_
#define lxl_nn_NN_H_

#include <iostream>
#include <random>
#include <nn/perceptron.h>
#include <dataLoader.h>

#define NO_RANDOMIZATION // Only for testing the algorithm, we need the same results for each run to compare.
#define USE_BP_BETA

/**
 * Sigmoid
 * Simple logistic function, It is a smooth, S-shaped curve.
 * Input: [0,1], Output: [0,1]
 */
template<typename Float>
Float sigmoid(Float x, Float a = 1) {
    return (Float) 1 / ((Float) 1 + std::exp(-a * x));
}

#define logic sigmoid

typedef std::size_t uzi;

class nn {
public:
    std::random_device rd;
    std::mt19937 e2;
    std::vector<std::vector<perceptron *>> P;
    std::vector<float> weightBackup;
    uzi outputIndex;
    float eta;
    float alpha;
    uzi sourceSize;
    uzi targetSize;
    uzi inputSize;
    uzi outputSize;
    std::vector<uzi> model;

    nn(std::vector<uzi> model, const std::vector<std::vector<float>> &input,
       const std::vector<std::vector<float>> &output) {
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

        create();
        connect();

#ifdef NO_RANDOMIZATION
        e2.seed(1);
#else
        e2.seed(rd());
#endif
        std::normal_distribution<float> dist(0, 1);

        outputIndex = network.size() - 1;
        inputSize = model[0];
        outputSize = model[outputIndex];

        shadow.resize(outputIndex);
        deltaShadow.resize(outputIndex);

        for (uzi k = 0; k < outputIndex; k++) {
            shadow[k].resize(model[k + 1]);
            deltaShadow[k].resize(model[k + 1]);
        }
#ifdef USE_BP_BETA
        errorSumBP.resize(outputSize);
        pid_P=0.25f;
        pid_I=0.005f/(float)sourceSize;
        pid_D=1.f;// 0.5*2=1
#endif
        initEtaAlpha();
    }

    ~nn() {
        //delete(P);
    }

    void create();

    void connect();

    void randWeights();

    void feedForward();

    void backPropagate();

    void setIO(std::vector<float> input, std::vector<float> output);

    void train(uzi loopMax);

    void checkTrainingData();

    void checkTestData(MNISTData *testData);

    void loadWeights();

    void saveWeights();

    void initValues();

    void initEtaAlpha();

private:
    std::vector<std::vector<float>> source;
    std::vector<std::vector<float>> target;
    std::vector<std::vector<float>> network;
    std::vector<std::vector<float>> shadow;
    std::vector<std::vector<float>> deltaShadow;
#ifdef USE_BP_BETA
    std::vector<float> errorSumBP;
    float pid_P;
    float pid_I;
    float pid_D;
#endif
};

#endif // lxl_nn_NN_H_
