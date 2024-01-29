#include "nn/neuralNetwork.h"
#ifdef LEARNING_MNIST_DATA
    #include "test/MNISTData.h"
#endif

int main() {

    auto *mainClock = new lxl::Timer();
#ifdef LEARNING_MNIST_DATA
    MNISTData trainingData;
    MNISTData testData;

    auto *mainClock = new lxl::Timer();

    // Loading the MNIST data
    if (!trainingData.load(true) || !testData.load(false)) {
        std::cout << "Could not load the MNIST data!" << std::endl;
        return 1;
    }

    auto *network = new NeuralNetwork({784, 300, 10}, trainingData.input, trainingData.output);

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
#else
     auto *network = new NeuralNetwork({4, 300, 4}, "../data/sort4.txt");

     network->printInfo();

     network->train(1000);
#endif

    safeDelete (network);
    safeDelete (mainClock);
    return 0;
}
