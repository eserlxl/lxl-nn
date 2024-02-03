#include "nn/neuralNetwork.h"

void example_sort4() {
    auto *mainClock = new Timer();

    std::cout << "\nSorting 4 numbers from lowest to highest" << std::endl;
    std::cout << "----------------------------------------------\n" << std::endl;

    /// Data file name to train
    std::string dataFile = "../data/sort4.txt";

    /// Define a library file name to save and load the network after training
    std::string libFile = "../test/lib/sort4.gz";

    /// Creating a new neural network to train data from a given file
    NeuralNetwork *network;
    if (std::filesystem::exists(libFile)) {
        network = new NeuralNetwork(libFile);

        /// A brief information of the created network
        network->printNetworkInfo();

        network->loadDataFromFile(dataFile);

        /// Normalized Root Mean Square Error Percentage of the library
        std::cout << "Library NRMSE(%): " << network->calcNormRMSEPercentage() << "\n" << std::endl;
    } else {
        network = new NeuralNetwork({4, 30, 120, 30, 4}, dataFile);

        /// A brief information of the created network
        network->printNetworkInfo();

        /// Training the neural network
        network->train(5000);

        /// Saving network to a compressed library file
        network->save(libFile);

        /// Normalized Root Mean Square Error Percentage
        std::cout << "\nTraining NRMSE(%): " << network->calcNormRMSEPercentage() << "\n" << std::endl;
    }

    /// Checking Re-training
    network->train(1);

    /// Normalized Root Mean Square Error Percentage after Re-training
    std::cout << "\nRe-Training NRMSE(%): " << network->calcNormRMSEPercentage() << std::endl;

    /// Checking neural network with a given input
    matrixFloat1D inputVec = {4, 1, 3, 2};
    network->setInput(inputVec);
    std::cout << "\nInput: " << std::endl;
    print(inputVec);

    /// Running the trained neural network
    network->feedForward();

    /// Output of the neural network for the given input
    matrixFloat1D outputVec = network->getOutput();
    std::cout << "\nOutput: " << std::endl;
    print(outputVec);

    /// Sorting input vector to check the learning error
    sort(inputVec);

    /// Calculating RMS error
    matrixFloat1D tempVec;
    for (uzi i = 0; i < outputVec.size(); i++) {
        tempVec.push_back(outputVec[i] - inputVec[i]);
    }
    std::cout << "\nRMSE: " << rms(tempVec) << std::endl;

    /// Delete network class
    safeDelete(network);

    std::cout << "\nElapsed time: " << mainClock->getElapsedTime() << " s" << std::endl;

    safeDelete(mainClock);
}