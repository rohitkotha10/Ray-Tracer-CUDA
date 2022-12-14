#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "vec3.h"
#include "color.h"

namespace Tracer {
    class Ray {
    public:
        __device__ Ray(const vec3& origin, const vec3& direction) {
            this->origin = origin;
            this->direction = direction;
        }
        __device__ vec3 getOrigin() const { return this->origin; }
        __device__ vec3 getDirection() const { return this->direction; }
        __device__ vec3 at(float t) const { return origin + t * direction; }

        vec3 origin;
        vec3 direction;
    };
}  // namespace Tracer
