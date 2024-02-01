## lxl-nn - Adaptive Deep Neural Network Architecture

This project aims a simple C++ implementation of an adaptive multi-layered feedforward-backpropagation neural network model.

Main features:
* Adaptive learning
* Multiple hidden layers
* Improved convergence
* Easy design and usage

## Usage
Network definition: _{ Input Layer, Hidden Layer, Output Layer }_

_Example: {6,12,6} or {3,5,5,3} etc._

#### Creating Neural Network
Available options to create a new neural network:

<p><b>1.</b> Create from input and output vectors</p>
<code>auto *network = new NeuralNetwork({5,6,5}, inputVector, outputVector)</code><br><br>

<p><b>2.</b> Create from a given data file</p>
<code>auto *network = new NeuralNetwork({4 3,6,9,2}, dataFileName);</code><br><br>

<p><b>3.</b> Create from previously trained compressed library file</p>
<code>auto *network = new NeuralNetwork(libraryFileName);</code>

#### Training Neural Network
<p>Chooes a max loop count for training</p>
<code>network->train(maxLoopCount);</code>

#### Save & Load Neural Network
Save network to a compressed library file for further usage.

<code>network->save(libraryFileName);</code>

Load network from the library

<code>network->load(libraryFileName);</code>

#### Check Neural Network

You can use setInput function to check a single input vector.

<code>network->setInput(inputVec);</code>

Get output using feedForward function.

<code>network->feedForward();</code>

<code>print(network->getOutput());</code>

#### Macro Definitions

_BP_BELLMAN_OPT_ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Uses Bellman's optimality to improve convergence

_BP_USE_BIAS_ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Use biases to train network quickly

_ADAPTIVE_LEARNING_ &nbsp;&nbsp;&nbsp;&nbsp;Activates an adaptive training algorithm to boost convergence

_ANALYSE_TRAINING_ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Detailed training output

_NO_RANDOMIZATION_ &nbsp;&nbsp;&nbsp;Disables randomization before starting to train

_LEARNING_MNIST_DATA_ Testing for MNIST data only

## Installation
You need to obtain [lxl](https://github.com/eserlxl/lxl) library before compilation.

Use CMAKE for compiling, unzip the Data.zip file for the required training and testing data.

## Testing
The MNIST database was used for testing the algorithm. 

More information about MNIST database and recent files can be obtained from [here](http://yann.lecun.com/exdb/mnist/).

## Bugs / TODO
* Prediction function will be implemented.
* Convergence problems for non-linear activation functions other than sigmoid.

## License
This project is licensed under the [GPLv3](LICENSE).
