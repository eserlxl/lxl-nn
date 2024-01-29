#include "nn/neuralNetwork.h"
#include "test/MNISTData.h"

int main() {

    MNISTData trainingData;
    MNISTData testData;

    auto *mainClock = new lxl::Timer();

    // Loading the MNIST data
    if (!trainingData.load(true) || !testData.load(false)) {
        std::cout << "Could not load the MNIST data!" << std::endl;
        return 1;
    }

    auto *network = new NeuralNetwork({784, 60, 10}, trainingData.input, trainingData.output);

    network->printInfo();

    network->train(100, &testData);

#ifndef ANALYSE_TRAINING
    TestResult trainingResult = network->checkTrainingData();
    TestResult testResult = network->checkTestData(&testData);
    std::cout<<"Error: "<<trainingResult.errorPercentage<<"%/"<<testResult.errorPercentage<<"% Time: "<<mainClock->getElapsedTime()<<" s"<<std::endl;
#else
    network->checkTestData(&testData);
    std::cout << std::endl << "Total time: " << mainClock->getElapsedTime() << " seconds" << std::endl;
#endif
    safeDelete (network);
    safeDelete (mainClock);
    return 0;
}
