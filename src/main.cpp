#include <nn.h>
#include "dataLoader.h"

int main() {

    MNISTData trainingData;
    MNISTData testData;

    auto *mainClock = new Timer();

    // Loading the MNIST data
    if (!trainingData.load(true) || !testData.load(false)) {
        std::cout << "Could not load the MNIST data!" << std::endl;
        return 1;
    }

    auto *NN = new nn({784, 10, 10}, trainingData.input, trainingData.output);

    NN->printInfo();

    NN->train(100, &testData);

#ifndef ANALYSE_TRAINING
    TestResult trainingResult = NN->checkTrainingData();
    TestResult testResult = NN->checkTestData(&testData);
    std::cout<<"Error: "<<trainingResult.errorPercentage<<"%/"<<testResult.errorPercentage<<"% Time: "<<mainClock->getElapsedTime()<<" s"<<std::endl;
#else
    NN->checkTestData(&testData);
    std::cout << std::endl << "Total time: " << mainClock->getElapsedTime() << " seconds" << std::endl;
#endif
    delete (NN);
    delete (mainClock);
    return 0;
}