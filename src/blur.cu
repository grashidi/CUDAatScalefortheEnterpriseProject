#include "blur.h"

namespace image_processing {

  double ApplyGaussianBlurNpp(const std::vector<unsigned char>& h_input, std::vector<unsigned char>& h_output, 
                        int width, int height, int kernel_size) {
    size_t size_to_copy = width * height * sizeof(Npp8u);

    Npp8u* d_input = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_input), size_to_copy);
    if (err != cudaSuccess) {
      std::cerr << "Failed to allocate device memory for d_input" << std::endl;
      return 0.0;
    }

    err = cudaMemcpy(d_input, h_input.data(), size_to_copy, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      cudaFree(d_input);
      std::cerr << "Failed to copy h_input data to d_input" << std::endl;
      return 0.0;
    }

    Npp8u* d_output = nullptr;
    err = cudaMalloc(reinterpret_cast<void**>(&d_output), size_to_copy);
    if (err != cudaSuccess) {
      cudaFree(d_input);
      std::cerr << "Failed to allocate device memory for d_output" << std::endl;
      return 0.0;
    }

    NppiSize image_size = { width, height };
    NppiMaskSize mask_size;
    switch (kernel_size) {
      case 3: mask_size = NPP_MASK_SIZE_3_X_3; break;
      case 5: mask_size = NPP_MASK_SIZE_5_X_5; break;
      case 7: mask_size = NPP_MASK_SIZE_7_X_7; break;
      default:
        std::cerr << "Unsupported kernel size. Use 3, 5, or 7." << std::endl;
        return 0.0;
    }

    auto start = std::chrono::high_resolution_clock::now();
    NppStatus status = nppiFilterGauss_8u_C1R(d_input, width, d_output, width, image_size, mask_size);
    auto end = std::chrono::high_resolution_clock::now();

    if (status != NPP_SUCCESS) {
      std::cerr << "NPP Gaussian blur failed! Error code: " << status << std::endl;
      cudaFree(d_input);
      cudaFree(d_output);
      return 0.0;
    }

    std::chrono::duration<double> elapsed = end - start;

    err = cudaMemcpy(h_output.data(), d_output, size_to_copy, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      std::cerr << "Failed to copy d_output data to h_output" << std::endl;
      cudaFree(d_input);
      cudaFree(d_output);
      return 0.0;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    
    return elapsed.count();
  }

  void GenerateGaussianKernel(std::vector<float>& kernel, int size, float sigma) {
      int half = size / 2;
      float sum = 0.0f;

      for (int y = -half; y <= half; y++) {
          for (int x = -half; x <= half; x++) {
              float value = expf(-(x * x + y * y) / (2 * sigma * sigma));
              kernel[(y + half) * size + (x + half)] = value;
              sum += value;
          }
      }

      // Normalize the kernel
      for (float& val : kernel) {
          val /= sum;
      }
  }

  // CUDA Gaussian blur kernel function.
  __global__ void GaussianBlurKernel(const unsigned char* image, 
                                  unsigned char* result, 
                                  uint32_t width, 
                                  uint32_t height, 
                                  const float* kernel, 
                                  int kernel_size) {
      int x = blockIdx.x * blockDim.x + threadIdx.x;
      int y = blockIdx.y * blockDim.y + threadIdx.y;
      int half_kernel = kernel_size / 2;

      if (x < width && y < height) {
          float sum = 0.0f;

          // Apply Gaussian kernel
          for (int ky = -half_kernel; ky <= half_kernel; ky++) {
          for (int kx = -half_kernel; kx <= half_kernel; kx++) {
              int ix = min(max(x + kx, 0), width - 1);  
              int iy = min(max(y + ky, 0), height - 1);
              sum += static_cast<float>(image[iy * width + ix]) *
                  kernel[(ky + half_kernel) * kernel_size + (kx + half_kernel)];
          }
          }

          // Clamp and store result
          result[y * width + x] = static_cast<unsigned char>(
              min(max(static_cast<int>(roundf(sum)), 0), 255));
      }
  }

  double ApplyGaussianBlurCuda(const std::vector<unsigned char>& image, 
                      std::vector<unsigned char>& result, 
                      uint32_t width, 
                      uint32_t height, 
                      const std::vector<float>& kernel, 
                      int kernel_size) {
      unsigned char* d_image = nullptr;
      unsigned char* d_result = nullptr;
      float* d_kernel = nullptr;

      size_t image_size = width * height * sizeof(unsigned char);
      size_t kernel_size_bytes = kernel_size * kernel_size * sizeof(float);

      // Allocate GPU memory
      cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_image), image_size);
      if (err != cudaSuccess) {
          std::cerr << "CUDA malloc failed for d_image: " << cudaGetErrorString(err) << std::endl;
          return 0.0;
      }

      err = cudaMalloc(reinterpret_cast<void**>(&d_result), image_size);
      if (err != cudaSuccess) {
          std::cerr << "CUDA malloc failed for d_result: " << cudaGetErrorString(err) << std::endl;
          cudaFree(d_image);
          return 0.0;
      }

      err = cudaMalloc(reinterpret_cast<void**>(&d_kernel), kernel_size_bytes);
      if (err != cudaSuccess) {
          std::cerr << "CUDA malloc failed for d_kernel: " << cudaGetErrorString(err) << std::endl;
          cudaFree(d_image);
          cudaFree(d_result);
          return 0.0;
      }

      // Copy data to device
      err = cudaMemcpy(d_image, image.data(), image_size, cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
          std::cerr << "CUDA memcpy failed for d_image: " << cudaGetErrorString(err) << std::endl;
          cudaFree(d_image);
          cudaFree(d_result);
          cudaFree(d_kernel);
          return 0.0;
      }

      err = cudaMemcpy(d_kernel, kernel.data(), kernel_size_bytes, cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
          std::cerr << "CUDA memcpy failed for d_kernel: " << cudaGetErrorString(err) << std::endl;
          cudaFree(d_image);
          cudaFree(d_result);
          cudaFree(d_kernel);
          return 0.0;
      }

      // Launch CUDA kernel
      dim3 block_dim(kBlockSize, kBlockSize);
      dim3 grid_dim((width + kBlockSize - 1) / kBlockSize, (height + kBlockSize - 1) / kBlockSize);
      
      auto start = std::chrono::high_resolution_clock::now();
      GaussianBlurKernel<<<grid_dim, block_dim>>>(d_image, d_result, width, height, d_kernel, kernel_size);
      auto end = std::chrono::high_resolution_clock::now();
      
      err = cudaGetLastError();
      if (err != cudaSuccess) {
          std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
          cudaFree(d_image);
          cudaFree(d_result);
          cudaFree(d_kernel);
          return 0.0;
      }

      cudaDeviceSynchronize();

      std::chrono::duration<double> elapsed = end - start;

      // Copy result back to host
      err = cudaMemcpy(result.data(), d_result, image_size, cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
          std::cerr << "CUDA memcpy failed for d_result: " << cudaGetErrorString(err) << std::endl;
          cudaFree(d_image);
          cudaFree(d_result);
          cudaFree(d_kernel);
          return 0.0;
      }

      // Free GPU memory
      cudaFree(d_image);
      cudaFree(d_result);
      cudaFree(d_kernel);
      
      return elapsed.count();
  }

}  // namespace image_processing 