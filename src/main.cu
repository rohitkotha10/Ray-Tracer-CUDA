#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vec3.h"
#include "utils.h"
#include "camera.h"
#include "hittable.h"
#include "color.h"
#include "ray.h"
// #include "hittable.h"
// #include "camera.h"

using namespace std;
using namespace Tracer;

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '"
                  << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void camInitialize(Camera **cam) {
    *cam = new Camera();
}

__global__ void createWorld(Hittable **d_list, Hittable **d_world) {
    *(d_list) = new Sphere(vec3(0, 0, -1), 0.5);
    *(d_list + 1) = new Sphere(vec3(0, -100.5, -1), 100);
    *d_world = new HittableList(d_list, 2);
}

__global__ void freeWorldCam(Hittable **d_list, Hittable **d_world, Camera **cam) {
    delete *(d_list);
    delete *(d_list + 1);
    delete *d_world;
    delete *cam;
}

__device__ vec3 color(const Ray &r, Hittable **world) {
    HitRecord rec;
    if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
        return 0.5f * vec3(rec.normal.x + 1.0f, rec.normal.y + 1.0f, rec.normal.z + 1.0f);
    } else {
        vec3 unit_direction = normalize(r.getDirection());
        float t = 0.5f * (unit_direction.y + 1.0f);
        return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
    }
    vec3 unit_direction = normalize(r.getDirection());
    float t = 0.5f * (unit_direction.y + 1.0f);
    return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

__global__ void render(float *fb, int width, int height, Camera** cam, Hittable **world) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= width) || (y >= height)) return;
    float u = (float)x / (float)width;
    float v = (float)y / (float)height;
    Ray r = (*cam)->getRay(u, v);
    vec3 col = color(r, world);

    int pixel_index = (y * width + x) * 3;
    fb[pixel_index + 0] = col.x;
    fb[pixel_index + 1] = col.y;
    fb[pixel_index + 2] = col.z;
}

int main() {
    Timer t1;
    t1.start("Image Generation");
    float aspectRatio = 16.0f / 9.0f;
    int width = 640;
    int height = ((float)width / aspectRatio);
    int samples = 8;
    int maxDepth = 32;

    Window cur(height, width);

    int num_pixels = width * height;
    size_t fb_size = 3 * num_pixels * sizeof(float);

    float *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    Camera **cam;
    checkCudaErrors(cudaMalloc((void **)&cam, sizeof(Camera *)));

    camInitialize<<<1, 1>>>(cam);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Hittable **d_list;
    Hittable **d_world;

    checkCudaErrors(cudaMalloc((void **)&d_list, 2 * sizeof(Hittable *)));
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(Hittable *)));

    createWorld<<<1, 1>>>(d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    int tx = 8;
    int ty = 8;
    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);
    render<<<blocks, threads>>>(fb, width, height, cam, d_world);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            vec3 temp(0.0f, 0.0f, 0.0f);
            size_t pixel_index = (y * width + x) * 3;
            float r = fb[pixel_index + 0];
            float g = fb[pixel_index + 1];
            float b = fb[pixel_index + 2];
            cur.writePixel(x, y, vec3(r, g, b), 1);
        }
    }

    t1.display();
    cur.saveWindow("Adv");

    checkCudaErrors(cudaDeviceSynchronize());
    freeWorldCam<<<1, 1>>>(d_list, d_world, cam);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(cam));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(fb));
}