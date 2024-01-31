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

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED COMPONENTS iostreams)

include_directories(${Boost_INCLUDE_DIRS})

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

target_link_libraries(lxl-nn lxl ${Boost_LIBRARIES})