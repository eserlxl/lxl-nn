#include "nn/neuralNetwork.h"
int main() {

    auto *mainClock = new Timer();

    std::cout<<"\nSorting 4 numbers from lowest to highest"<<std::endl;
    std::cout<<"----------------------------------------------\n"<<std::endl;

    /// Creating a new neural network to train data from a given file
    auto *network = new NeuralNetwork({4, 30,120,30, 4}, "../data/sort4.txt");

    /// A brief information of the created network
    network->printNetworkInfo();

    /// Training the neural network
    network->train(5000);

    /// Checking neural network with a given input
    std::vector<float> inputVec = {4, 1, 3, 2};
    network->setInput(inputVec);
    std::cout<<"Input: "<<std::endl;
    print(inputVec);

    /// Running the trained neural network
    network->feedForward();

    /// Output of the neural network for the given input
    std::vector<float> outputVec = network->getOutput();
    std::cout<<"\nOutput: "<<std::endl;
    print(outputVec);

    /// Calculating RMS error
    std::vector<float> tempVec;
    for (uzi i = 0; i < outputVec.size(); i++) {
        tempVec.push_back(outputVec[i] - inputVec[i]);
    }
    std::cout << "\nRMSE: " << rms(tempVec) << std::endl;

    std::string libFile = "../test/lib/sort4.gz";

    /// Saving network to a compressed library file
    network->save(libFile);

    /// Loading from library, same class
    network->load(libFile);

    network->setInput(inputVec);
    network->feedForward();

    std::cout<<"\nOutput from library (same class): "<<std::endl;
    print(network->getOutput());

    safeDelete(network);
    /// End of loading from library

    /// Loading from library, completely a new class
    auto *networkFromLibrary = new NeuralNetwork(libFile);

    networkFromLibrary->setInput(inputVec);
    networkFromLibrary->feedForward();

    std::cout<<"\nOutput from library (new class): "<<std::endl;
    print(networkFromLibrary->getOutput());

    safeDelete(networkFromLibrary);
    /// End of loading from library

    std::cout << "\nElapsed time: " << mainClock->getElapsedTime() << " s" << std::endl;

    safeDelete(mainClock);
    return 0;
}
