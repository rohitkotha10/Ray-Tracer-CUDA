#pragma once
#include <vector>
#include <memory>

#include "vec3.h"
#include "ray.h"

namespace Tracer {
    struct HitRecord {
        vec3 point;
        vec3 normal;
        float t;
        bool frontFace;
        inline void setFaceNormal(const Ray& ray, const vec3& outwardNormal) {
            frontFace = dot(ray.getDirection(), outwardNormal) < 0;
            if (frontFace == false)
                normal = -outwardNormal;
            else
                normal = outwardNormal;
        }
    };

    class Hittable {
    public:
        virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const = 0;
    };

    class Sphere: public Hittable {
    public:
        Sphere(vec3 cen, float r) {
            this->center = cen;
            this->radius = r;
        }

        virtual bool hit(const Ray& ray, float tMin, float tMax, HitRecord& rec) const override {
            vec3 oc = ray.getOrigin() - center;
            float a = dot(ray.getDirection(), ray.getDirection());
            float b = 2.0 * dot(oc, ray.getDirection());
            float c = dot(oc, oc) - radius * radius;
            float discriminant = b * b - 4 * a * c;
            if (discriminant < 0) { return false; }
            float root = (-b - sqrt(discriminant)) / (2.0 * a);

            if (root < tMin || tMax < root) {
                root = (-b + sqrt(discriminant)) / (2.0 * a);
                if (root < tMin || tMax < root) return false;
            }
            rec.t = root;
            rec.point = ray.at(rec.t);
            vec3 outwardNormal = (rec.point - center) / radius;
            rec.setFaceNormal(ray, outwardNormal);

            return true;
        }

    private:
        vec3 center;
        float radius;
    };

    class HittableList: public Hittable {
    public:
        HittableList() {}
        HittableList(std::shared_ptr<Hittable> object) { add(object); }

        void clear() { objects.clear(); }
        void add(std::shared_ptr<Hittable> object) { objects.push_back(object); }
        virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const override {
            HitRecord tempRec;
            bool hitAnything = false;
            float closestSoFar = tMax;

            for (const auto& object: objects) {
                if (object->hit(r, tMin, closestSoFar, tempRec)) {
                    hitAnything = true;
                    closestSoFar = tempRec.t;
                    rec = tempRec;
                }
            }

            return hitAnything;
        }

    private:
        std::vector<std::shared_ptr<Hittable>> objects;
    };
}  // namespace Tracer
