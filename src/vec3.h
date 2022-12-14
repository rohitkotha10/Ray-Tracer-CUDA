#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>

namespace Tracer {
    class vec3 {
    public:
        __host__ __device__ vec3() {}
        __host__ __device__ vec3(float x, float y, float z) {
            this->x = x;
            this->y = y;
            this->z = z;
        }
        __host__ __device__ inline float xval() const { return x; }
        __host__ __device__ inline float yval() const { return y; }
        __host__ __device__ inline float zval() const { return z; }
        __host__ __device__ inline float length() const { return sqrt(x * x + y * y + z * z); }
        __host__ __device__ inline float lengthSquared() const { return x * x + y * y + z * z; }

        __host__ __device__ inline const vec3 &operator+() const { return *this; }
        __host__ __device__ vec3 operator-() const { return vec3(-x, -y, -z); }

        float x;
        float y;
        float z;
    };

    __host__ __device__ vec3 operator+(const vec3 &v1, const vec3 &v2) {
        return vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
    }

    __host__ __device__ vec3 operator-(const vec3 &v1, const vec3 &v2) {
        return vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
    }

    __host__ __device__ inline vec3 operator*(float t, const vec3 &v) {
        return vec3(t * v.x, t * v.y, t * v.z);
    }

    __host__ __device__ inline vec3 operator/(vec3 v, float t) {
        return vec3(v.x / t, v.y / t, v.z / t);
    }

    __host__ __device__ inline vec3 operator*(const vec3 &v, float t) {
        return vec3(t * v.x, t * v.y, t * v.z);
    }
    __host__ __device__ float dot(const vec3 &v1, const vec3 &v2) {
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    }

    __host__ __device__ inline vec3 cross(const vec3 &v1, const vec3 &v2) {
        return vec3((v1.y * v2.z - v1.z * v2.y), (-(v1.x * v2.z - v1.z * v2.x)), (v1.x * v2.y - v1.y * v2.x));
    }

    __host__ __device__ vec3 normalize(vec3 v) {
        float k = 1.0 / sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        v.x *= k;
        v.y *= k;
        v.z *= k;
        return v;
    }
}  // namespace Tracer
