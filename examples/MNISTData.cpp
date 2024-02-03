#include "nn/neuralNetwork.h"
#include "test/MNISTData.h"

void example_MNISTData() {
    auto *mainClock = new Timer();

    float minNRMSEPercentage = 1e6;

    std::cout << "\nLearning handwritten digits from MNIST Data" << std::endl;
    std::cout << "----------------------------------------------\n" << std::endl;

    /// Data file names are defined in MNISTData.h
    MNISTData trainingData;
    MNISTData testData;

    /// Loading the MNIST data
    if (!trainingData.load(true) || !testData.load(false)) {
        std::cout << "Could not load the MNIST data!" << std::endl;
        return;
    }

    /// Define a library file name to save and load the network after training
    std::string libFile = "../test/lib/MNIST.gz";

    /// Creating a new neural network to train data from a given file
    NeuralNetwork *network;
    if (std::filesystem::exists(libFile)) {
        network = new NeuralNetwork(libFile);

        /// A brief information of the created network
        network->printNetworkInfo();

        network->normIO(trainingData.input, trainingData.output);

        /// Normalized Root Mean Square Error Percentage of the library
        minNRMSEPercentage = network->NRMSEPercentage;
        std::cout << "Library NRMSE(%): " << minNRMSEPercentage << "\n" << std::endl;

        /// Get error percentage
        std::cout << "Training error (library): " << network->checkBinaryOutputData() * 100.f << "%\n" << std::endl;
    } else {
        network = new NeuralNetwork({784, 300, 10});

        network->normIO(trainingData.input, trainingData.output);

        /// A brief information of the created network
        network->printNetworkInfo();

        /// Training the neural network
        network->train(100);

        /// Saving network to a compressed library file
        network->save(libFile);

        /// Normalized Root Mean Square Error Percentage
        std::cout << "\nTraining NRMSE(%): " << network->NRMSEPercentage << "\n" << std::endl;

#ifdef BINARY_OUTPUT_DATA
        /// Get error percentage
        std::cout << "Training error: " << network->binaryDataErrorPercentage << "%\n" << std::endl;
#endif
    }

    /// Checking Re-training
    network->train(1);

    /// Normalized Root Mean Square Error Percentage after Re-training
    std::cout << "\nRe-Training NRMSE(%): " << network->NRMSEPercentage << std::endl;

    /// Loading Test data to network. Check: http://yann.lecun.com/exdb/mnist/ for more information.
    network->normIO(testData.input, testData.output);

    /// Normalized Root Mean Square Error Percentage of the Test Data
    std::cout << "\nTest Data NRMSE(%): " << network->calcNormRMSEPercentage() << std::endl;

    /// Get error percentage
    std::cout << "Test Data error: " << network->checkBinaryOutputData() * 100.f << "%\n" << std::endl;

    /// Saving network to a compressed library file
    if (network->NRMSEPercentage < minNRMSEPercentage) {
        network->save(libFile);
    }

    /// Delete network class
    safeDelete(network);

    std::cout << "\nElapsed time: " << mainClock->getElapsedTime() << " s" << std::endl;

    safeDelete(mainClock);
}
