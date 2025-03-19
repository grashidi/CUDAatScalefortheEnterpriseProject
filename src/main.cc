#include "utils.h"
#include "blur.h"

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <kernel_size> <image_path>" << std::endl;
    return 1;
  }

  int kernel_size = std::stoi(argv[1]);
  std::string tmp = argv[2];
  const char* image_path = tmp.c_str();

  std::vector<unsigned char> image;
  uint32_t width, height;

  if (!image_utils::LoadTiff(image_path, image, width, height)) {
    return -1;
  }

  std::vector<unsigned char> blurred_image(width * height);
  double npp_time = image_processing::ApplyGaussianBlurNpp(image, blurred_image, width, height, kernel_size);

  std::string path = "images/aerials_blurred_npp";
  size_t dotPos = tmp.find_last_of('.');
  size_t slashPos = tmp.find_last_of('/');
  std::string out_file = path + tmp.substr(slashPos);
  image_utils::SaveTiff(out_file.c_str(), blurred_image, width, height);

  std::vector<float> kernel(kernel_size * kernel_size);
  image_processing::GenerateGaussianKernel(kernel, kernel_size, 5.0f); // Using sigma = 5.0
  double cuda_time = image_processing::ApplyGaussianBlurCuda(image, blurred_image, width, height, kernel, kernel_size);

  path = "images/aerials_blurred_cuda";
  dotPos = tmp.find_last_of('.');
  slashPos = tmp.find_last_of('/');
  out_file = path + tmp.substr(slashPos);
  image_utils::SaveTiff(out_file.c_str(), blurred_image, width, height);

  std::cout << kernel_size << "\t" << npp_time << "\t" << cuda_time << std::endl;

  return 0;
}