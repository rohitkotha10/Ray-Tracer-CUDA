#pragma once
#include <vector>
#include <string>
#include <iostream>

#include "vec3.h"

#include "../vendor/stb_image/stb_image.h"
#include "../vendor/stb_image/stb_image_write.h"

namespace Tracer {
    class Window {
    public:
        Window(int height, int width) {
            this->height = height;
            this->width = width;
            data.resize(width);
            for (int i = 0; i < width; i++) { data[i].resize(height); }
        }

        void writePixel(int x, int y, const vec3 &col) { data[x][y] = col; }

        void saveWindow(std::string filename) {
            int imgSize = height * width * 3;

            unsigned char *img = new unsigned char[imgSize];
            unsigned char *start = img;

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int cur = ((height - y - 1) * width * 3) + x * 3;

                    float r = data[x][y].x;
                    float g = data[x][y].y;
                    float b = data[x][y].z;
                    *(start + cur) = (uint8_t)(255.99f * r);
                    *(start + 1 + cur) = (uint8_t)(255.99f * g);
                    *(start + 2 + cur) = (uint8_t)(255.99f * b);
                }
            }
            std::string name = filename + ".jpg";
            stbi_write_jpg(name.c_str(), width, height, 3, img, 100);
            stbi_image_free(img);

            std::cout << std::endl << "Image Generated to " << name << std::endl;
        }

    private:
        int height;
        int width;
        std::vector<std::vector<vec3>> data;
    };
}  // namespace Tracer
