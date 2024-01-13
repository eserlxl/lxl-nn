## lxl-nn - A simple Neural Network architecture

A feedforward-backpropagation neural network model is used with small modifications.
Multiple hidden layers are supported.

## Info
Network definition: _{ Input Layer, Hidden Layer, Output Layer }_

_Example: {6,12,6} or {3,5,5,3} etc._

## Modifications
* Backpropagation weight updating improvements
* PID design for error feeding in backpropagation.

## Bugs / TODO
* Multiple hidden layer design problem: hiddenLayerNum < outputLayerNum gives size error
* Backpropagation algorithm should be updated for multiple hidden layers.

## Testing
The MNIST database was used for testing the algorithm. 
More information about MNIST database and recent files can be obtained from [here](http://yann.lecun.com/exdb/mnist/).

## Performance
MNIST database tests were conducted on 12th Gen Intel(R) Core(TM) i7-12700H CPU (single thread only).

**Training/Test data error:** 0.005%/1.69%, Hidden Layer: {300}, loopMax=20, Time: 1808.21 seconds

Please check results.txt for additional information.

## Usage
Use CMAKE for compiling, unzip the Data.zip file for the required training and testing data.

## License
This project is licensed under the [GPLv3](LICENSE).
