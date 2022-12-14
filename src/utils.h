#pragma once
#include "vec3.h"

#include <chrono>
#include <string>
#include <random>
#include <iostream>

namespace Tracer {
    const float infinity = std::numeric_limits<float>::infinity();
    const float pi = 3.14159265;

    float clamp(float x, float min, float max) {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }
    float getRandomFloat(float min, float max) {
        static std::uniform_real_distribution<float> distribution(min, max);
        static std::mt19937 generator;
        return distribution(generator);
    }

    vec3 getRandomVec(float min, float max) {
        return vec3(getRandomFloat(min, max), getRandomFloat(min, max), getRandomFloat(min, max));
    }

    vec3 getRandomInUnitSphere() {
        while (true) {
            vec3 temp = getRandomVec(-1.0f, 1.0f);
            if (temp.length() <= 1) return temp;
        }
        std::cout << "Not found unit sphere" << std::endl;
        return vec3(-1.0f);
    }

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
