#ifndef lxl_nn_TIMER_H
#define lxl_nn_TIMER_H

#include <iostream>
#include <chrono>

typedef std::chrono::high_resolution_clock::time_point timer;

class Timer {
    timer T;

public:
    explicit Timer() {
        initTimer();
    }
    void initTimer() {
        T=std::chrono::high_resolution_clock::now();
    }

    double getElapsedTime() {
        timer N = std::chrono::high_resolution_clock::now();
        return (std::chrono::duration_cast<std::chrono::nanoseconds>(N - T).count()) / 1.e9;
    }


    ~Timer() {
    }


};

#endif //lxl_nn_TIMER_H