#pragma once
#include <vector>
#include <string>

#include <iostream>
#include "vec3.h"
#include "utils.h"

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

        void writePixel(int x, int y, const vec3 &col, int samples) {
            vec3 cur = col;

            float scale = 1.0 / (float)samples;
            cur.x = sqrt(scale * cur.x);
            cur.y = sqrt(scale * cur.y);
            cur.z = sqrt(scale * cur.z);

            data[x][y] = cur;
        }

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
                    *(start + cur) = (uint8_t)(255.99f * clamp(r, 0.0f, 0.999f));
                    *(start + 1 + cur) = (uint8_t)(255.99f * clamp(g, 0.0f, 0.999f));
                    *(start + 2 + cur) = (uint8_t)(255.99f * clamp(b, 0.0f, 0.999f));
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
