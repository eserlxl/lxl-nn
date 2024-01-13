#include <nn.h>
#include "dataLoader.h"
#include "timer.h"

int main() {
    MNISTData trainingData;
    MNISTData testData;

    Timer timer("The training time:  ");

    // Loading the MNIST data
    if (!trainingData.load(true) || !testData.load(false)) {
        printf("Could not load the MNIST data!\n");
        return 1;
    }

    auto *NN = new nn({784, 300, 10}, trainingData.input, trainingData.output);
    NN->train(20);
    NN->checkTrainingData();
    NN->checkTestData(&testData);
    delete (NN);
    return 0;
}