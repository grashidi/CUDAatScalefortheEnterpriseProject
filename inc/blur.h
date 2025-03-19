#ifndef PROJECT_BLUR_H_
#define PROJECT_BLUR_H_

#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <cmath>
#include <npp.h>


const int kBlockSize = 16;

namespace image_processing {
    // Applies a Gaussian blur to the input image using NPP.
    double ApplyGaussianBlurNpp(const std::vector<unsigned char>& h_input, 
                        std::vector<unsigned char>& h_output, 
                        int width, int height, int kernel_size);

    // Generates a Gaussian kernel.
    void GenerateGaussianKernel(std::vector<float>& kernel, int size, float sigma);


    // Host function declaration
    double ApplyGaussianBlurCuda(const std::vector<unsigned char>& image, 
                            std::vector<unsigned char>& result, 
                            uint32_t width, 
                            uint32_t height, 
                            const std::vector<float>& kernel, 
                            int kernel_size); 
                            
    __global__ void GaussianBlurKernel(const unsigned char* image, 
                                    unsigned char* result, 
                                    uint32_t width, 
                                    uint32_t height, 
                                    const float* kernel, 
                                    int kernel_size);

}  // namespace image_processing

#endif  // PROJECT_BLUR_H_