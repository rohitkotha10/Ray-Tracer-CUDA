#pragma once
#include "vec3.h"
#include "ray.h"

namespace Tracer {
    class Camera {
    public:
        Camera() {
            float aspectRatio = 16.0f / 9.0f;

            float viewportHeight = 2.0f;
            float viewportWidth = aspectRatio * viewportHeight;
            float focal = 1.0f;

            origin = vec3(0.0f, 0.0f, 0.0f);
            horizontal = vec3(viewportWidth, 0.0f, 0.0f);
            vertical = vec3(0.0f, viewportHeight, 0.0f);
            lowLeftCorner =
                origin - horizontal / 2.0f - vertical / 2.0f - vec3(0.0f, 0.0f, focal);  // low left corner
        }
        Ray getRay(float u, float v) { return Ray(origin, lowLeftCorner + u * horizontal + v * vertical - origin); }

    private:
        vec3 origin;
        vec3 lowLeftCorner;
        vec3 horizontal;
        vec3 vertical;
    };
}  // namespace Tracer
