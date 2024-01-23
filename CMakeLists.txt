cmake_minimum_required(VERSION 3.16)
project(lxl-nn)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -gdwarf-3")#-Ofast
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -gdwarf-3")

include_directories(include)

add_executable(lxl-nn
        include/nn/neuron.h
        include/nn/synapse.h
        include/nn.h
        include/timer.h
        src/nn/backPropagate.cpp
        src/nn/feedForward.cpp
        src/nn/init.cpp
        src/nn/nn.cpp
        src/nn/neuron.cpp
        src/nn/train.cpp
        src/nn/weights.cpp
        src/main.cpp include/test.h)