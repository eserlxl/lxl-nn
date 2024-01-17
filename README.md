## lxl-nn - A Simple Neural Network Architecture

A feedforward-backpropagation neural network model is used with small modifications.

Multiple hidden layers are supported. [MNIST database](http://yann.lecun.com/exdb/mnist/) is used for development process.

<style>
  table td {
  text-align: center;
    border: none !important;
  };
  td, th {
  text-align: center;
      border: none!important;
  }
</style>

<table>
  <tr>
    <th><img src="https://drive.google.com/uc?export=view&id=1nm3mMZdw_agfxkFInk1zcs4oMge3gNBg"/></th>
  </tr>
  <tr>
    <th>The minimum test data error for the MNIST database is 1.57%.</th>
  </tr>
</table>

## Modifications
* Backpropagation weight updating improvements
* PID design for error feeding in backpropagation.

## Bugs / TODO
* Multiple hidden layer design problem: hiddenLayerNum < outputLayerNum gives size error
* Backpropagation algorithm should be updated for multiple hidden layers.

## Testing
The MNIST database was used for testing the algorithm. 

More information about MNIST database and recent files can be obtained from [here](http://yann.lecun.com/exdb/mnist/).

Use _ANALYSE_TRAINING_ macro for a detailed output.

## Performance
MNIST database tests were conducted on a 12th Gen Intel(R) Core(TM) i7-12700H CPU (single thread only).

**Training/Test data error:** 0%/1.64%, Hidden Layer: {300}, loopMax=30, Time: 2653.99 seconds

Please check results.txt for additional information.

## Usage
Use CMAKE for compiling, unzip the Data.zip file for the required training and testing data.

Network definition: _{ Input Layer, Hidden Layer, Output Layer }_

_Example: {6,12,6} or {3,5,5,3} etc._

## License
This project is licensed under the [GPLv3](LICENSE).
