#pragma once
#include "vec3.h"

#include <chrono>
#include <string>
#include <random>
#include <iostream>

namespace Tracer {
    const float pi = 3.14159265;

    void printVec(vec3 cur) {
        std::cout << cur.x << ' ' << cur.y << ' ' << cur.z << std::endl;
    }

    class Timer {
    public:
        void start(std::string name) {
            this->name = name;
            startTime = std::chrono::high_resolution_clock::now();
        }

        void display() {
            curTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> dur = curTime - startTime;
            std::cout << name << " Completed in " << dur.count() << "s" << std::endl;
        }

    private:
        std::chrono::steady_clock::time_point startTime;
        std::chrono::steady_clock::time_point curTime;
        std::string name;
    };
}  // namespace Tracer
