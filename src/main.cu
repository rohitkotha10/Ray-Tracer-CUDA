#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vec3.h"
#include "utils.h"
#include "color.h"
#include "ray.h"
//#include "hittable.h"
//#include "camera.h"

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

__device__ vec3 color(const Ray &r) {
    vec3 unit_direction = normalize(r.getDirection());
    float t = 0.5f * (unit_direction.y + 1.0f);
    return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

__global__ void render(
    float *fb, int width, int height, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= width) || (y >= height)) return;
    int pixel_index = (y * width + x) * 3;
    float u = (float)x / (float)width;
    float v = (float)y / (float)height;
    Ray r(origin, lower_left_corner + u * horizontal + v * vertical);
    vec3 col = color(r);
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

    float viewportHeight = 2.0f;
    float viewportWidth = aspectRatio * viewportHeight;
    vec3 origin = vec3(0.0f, 0.0f, 0.0f);
    vec3 horizontal = vec3(viewportWidth, 0.0f, 0.0f);
    vec3 vertical = vec3(0.0f, viewportHeight, 0.0f);
    vec3 lowLeftCorner = origin - horizontal / 2.0f - vertical / 2.0f - vec3(0.0f, 0.0f, 1.0f);  // low left corner

    int tx = 8;
    int ty = 8;
    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);
    render<<<blocks, threads>>>(fb, width, height, lowLeftCorner, horizontal, vertical, origin);

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

    // int img_size = height * width * 3;

    // unsigned char *img = new unsigned char[img_size];
    // unsigned char *start = img;

    // for (int y = 0; y < height; y++) {
    //     for (int x = 0; x < width; x++) {
    //         int cur = ((height - y - 1) * width + x) * 3;

    //        size_t pixel_index = (y * width + x) * 3;
    //        float r = fb[pixel_index + 0];
    //        float g = fb[pixel_index + 1];
    //        float b = fb[pixel_index + 2];
    //        *(start + cur) = (uint8_t)(255.99 * r);
    //        *(start + 1 + cur) = (uint8_t)(255.99 * g);
    //        *(start + 2 + cur) = (uint8_t)(255.99 * b);
    //    }
    //}
    // stbi_write_jpg("gradient.jpg", width, height, 3, img, 100);

    checkCudaErrors(cudaFree(fb));
}