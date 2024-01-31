cmake_minimum_required(VERSION 3.16)
project(lxl-nn)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -fPIC -O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -fPIC -O0 -gdwarf-4")

include_directories(include)
include_directories(/usr/include/eigen3) # Path to Eigen library
include_directories(/usr/include/lxl) # Path to lxl library

link_libraries(lxl)

add_executable(lxl-nn
        include/nn/neuron.h
        include/nn/synapse.h
        include/nn/neuralNetwork.h
        include/test/analyse.h
        include/test/test.h
        include/test/testResult.h
        src/nn/backPropagate.cpp
        src/nn/feedForward.cpp
        src/nn/init.cpp
        src/nn/neuralNetwork.cpp
        src/nn/neuron.cpp
        src/nn/normIO.cpp
        src/nn/train.cpp
        src/nn/weights.cpp
        src/main.cpp)

