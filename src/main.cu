#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>

// function to add the elements of two arrays
#define STB_IMAGE_IMPLEMENTATION
#include "../vendor/stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../vendor/stb_image/stb_image_write.h"

using namespace std;

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

__global__ void render(float *fb, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= width) || (y >= height)) return;
    int pixel_index = (y * width  + x)* 3;
    fb[pixel_index + 0] = float(x) / width;
    fb[pixel_index + 1] = float(y) / height;
    fb[pixel_index + 2] = 0.2;
}

int main() {
    int width = 256;
    int height = 256;

    int num_pixels = width * height;
    size_t fb_size = 3 * num_pixels * sizeof(float);

    float *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    int tx = 8;
    int ty = 8;
    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);
    render<<<blocks, threads>>>(fb, width, height);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    int img_size = height * width * 3;

    unsigned char *img = new unsigned char[img_size];
    unsigned char *start = img;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int cur = ((height - y - 1) * width + x) * 3;

            size_t pixel_index = (y * width + x) * 3;
            float r = fb[pixel_index + 0];
            float g = fb[pixel_index + 1];
            float b = fb[pixel_index + 2];
            *(start + cur) = (uint8_t)(255.99 * r);
            *(start + 1 + cur) = (uint8_t)(255.99 * g);
            *(start + 2 + cur) = (uint8_t)(255.99 * b);
        }
    }
    stbi_write_jpg("gradient.jpg", width, height, 3, img, 100);

    checkCudaErrors(cudaFree(fb));
}