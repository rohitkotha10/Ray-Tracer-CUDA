#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "vec3.h"
#include "ray.h"

namespace Tracer {
    class HitRecord {
    public:
        vec3 point;
        vec3 normal;
        float t;
        bool frontFace;
        __device__ void setFaceNormal(const Ray& ray, const vec3& outwardNormal) {
            frontFace = dot(ray.getDirection(), outwardNormal) < 0;
            if (frontFace == false)
                normal = -outwardNormal;
            else
                normal = outwardNormal;
        }
    };

    class Hittable {
    public:
        __device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const = 0;
    };

    class Sphere: public Hittable {
    public:
        __device__ Sphere(vec3 cen, float r) : center(cen), radius(r) {}

        __device__ virtual bool hit(const Ray& ray, float tMin, float tMax, HitRecord& rec) const {
            vec3 oc = ray.getOrigin() - center;
            float a = dot(ray.getDirection(), ray.getDirection());
            float b = dot(oc, ray.getDirection());
            float c = dot(oc, oc) - radius * radius;
            float discriminant = b * b - a * c;
            if (discriminant > 0) {
                float temp = (-b - sqrt(discriminant)) / a;
                if (temp < tMax && temp > tMin) {
                    rec.t = temp;
                    rec.point = ray.at(rec.t);
                    rec.normal = (rec.point - center) / radius;
                    return true;
                }
                temp = (-b + sqrt(discriminant)) / a;
                if (temp < tMax && temp > tMin) {
                    rec.t = temp;
                    rec.point = ray.at(rec.t);
                    rec.normal = (rec.point - center) / radius;
                    return true;
                }
            }
            return false;
        }
        vec3 center;
        float radius;
    };

    class HittableList: public Hittable {
    public:
        __device__ HittableList() {}
        __device__ HittableList(Hittable** objectsList, int size) {
            objects = objectsList;
            listSize = size;
        }

        __device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const {
            HitRecord tempRec;
            bool hitAnything = false;
            float closestSoFar = tMax;

            for (int i = 0; i < listSize; i++) {
                if (objects[i]->hit(r, tMin, closestSoFar, tempRec)) {
                    hitAnything = true;
                    closestSoFar = tempRec.t;
                    rec = tempRec;
                }
            }
            return hitAnything;
        }

        Hittable** objects;
        int listSize;
    };
}  // namespace Tracer
