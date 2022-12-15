#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vec3.h"
#include "utils.h"
#include "camera.h"
#include "hittable.h"
#include "color.h"
#include "ray.h"

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

__global__ void createWorldCam(Hittable **d_list, Hittable **d_world, Camera **cam) {
    *cam = new Camera();
    *(d_list) = new Sphere(vec3(0, 0, -1), 0.5);
    *(d_list + 1) = new Sphere(vec3(0, -100.5, -1), 100);
    *d_world = new HittableList(d_list, 2);
}

__global__ void freeWorldCam(Hittable **d_list, Hittable **d_world, Camera **cam) {
    delete *cam;
    delete *(d_list);
    delete *(d_list + 1);
    delete *d_world;
}

#define RANDVEC3 \
    vec3(curand_uniform(localStateAddress), curand_uniform(localStateAddress), curand_uniform(localStateAddress))

__device__ vec3 randomInUnitSphere(curandState *localStateAddress) {
    vec3 temp;
    while (true) {
        temp = 2.0f * RANDVEC3 - vec3(1.0, 1.0, 1.0);
        if (temp.lengthSquared() < 1) return temp;
    }
    return vec3(0.1, 0.1, 0.1);
}

__device__ vec3 rayColor(const Ray &r, Hittable **world, curandState *localStateAddress, int maxDepth) {
    Ray curRay = r;
    float curAttenuation = 1.0f;
    for (int i = 0; i < maxDepth; i++) {
        HitRecord rec;
        if ((*world)->hit(curRay, 0.001f, FLT_MAX, rec)) {
            vec3 target = rec.point + rec.normal + randomInUnitSphere(localStateAddress);
            curAttenuation *= 0.5f;
            curRay = Ray(rec.point, target - rec.point);
        } else {
            vec3 unitVec = normalize(curRay.getDirection());
            float t = 0.5f * (unitVec.y + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return curAttenuation * c;
        }
    }
    return vec3(0.0, 0.0, 0.0);  // too many bounces, so a black shadow
}

__global__ void renderInit(int width, int height, curandState *randState) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= width) || (y >= height)) return;
    int pixel_index = y * width + x;

    curand_init(1984, pixel_index, 0, &randState[pixel_index]);
}

__global__ void render(
    vec3 *fb,
    int width,
    int height,
    Camera **cam,
    Hittable **world,
    int samples,
    int maxDepth,
    curandState *randState) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= width) || (y >= height)) return;
    int pixel_index = y * width + x;

    curandState localState = randState[pixel_index];
    vec3 col(0.0f, 0.0f, 0.0f);
    for (int s = 0; s < samples; s++) {
        float u = float(x + curand_uniform(&localState)) / float(width);
        float v = float(y + curand_uniform(&localState)) / float(height);
        Ray r = (*cam)->getRay(u, v);
        col = col + rayColor(r, world, &localState, maxDepth);
    }

    col = col / float(samples);
    col.x = sqrt(col.x);
    col.y = sqrt(col.y);
    col.z = sqrt(col.z);

    fb[pixel_index] = col;
}

int main() {
    Timer t1;
    t1.start("Image Generation");
    float aspectRatio = 16.0f / 9.0f;
    int width = 1280;
    int height = ((float)width / aspectRatio);
    int samples = 100;
    int maxDepth = 50;

    int tx = 8;
    int ty = 8;
    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);

    Window cur(height, width);

    int numPixels = width * height;

    Camera **cam;
    checkCudaErrors(cudaMalloc((void **)&cam, sizeof(Camera *)));

    Hittable **objectsList;
    checkCudaErrors(cudaMalloc((void **)&objectsList, 2 * sizeof(Hittable *)));

    Hittable **myWorld;
    checkCudaErrors(cudaMalloc((void **)&myWorld, sizeof(Hittable *)));

    createWorldCam<<<1, 1>>>(objectsList, myWorld, cam);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    curandState *randState;
    checkCudaErrors(cudaMalloc((void **)&randState, numPixels * sizeof(curandState)));

    renderInit<<<blocks, threads>>>(width, height, randState);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    size_t fb_size = numPixels * sizeof(vec3);
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    render<<<blocks, threads>>>(fb, width, height, cam, myWorld, samples, maxDepth, randState);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            vec3 temp(0.0f, 0.0f, 0.0f);
            size_t pixel_index = y * width + x;
            cur.writePixel(x, y, fb[pixel_index]);
        }
    }

    t1.display();
    cur.saveWindow("Diffuse");

    checkCudaErrors(cudaDeviceSynchronize());
    freeWorldCam<<<1, 1>>>(objectsList, myWorld, cam);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(cam));
    checkCudaErrors(cudaFree(objectsList));
    checkCudaErrors(cudaFree(myWorld));
    checkCudaErrors(cudaFree(fb));
}