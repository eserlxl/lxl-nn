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

    auto *NN = new nn({784,10,10},trainingData.input, trainingData.output);
    NN->train(10);
    NN->checkTrainingData();
    NN->checkTestData(&testData);
    //safeDelete(NN);
    return 0;
}