#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <stdexcept>

#include "cudaImage.h"
#include "cudaSift.h"

namespace py = pybind11;

py::tuple extract_sift(
    const py::array_t<float, py::array::c_style | py::array::forcecast> &image,
    int num_octaves,
    float init_blur,
    float thresh,
    float lowest_scale,
    bool scale_up,
    int max_pts,
    int dev_num) {
  py::buffer_info buf = image.request();
  if (buf.ndim != 2) {
    throw std::invalid_argument("image must be a 2D float32 array (H, W)");
  }

  const int height = static_cast<int>(buf.shape[0]);
  const int width = static_cast<int>(buf.shape[1]);
  if (height <= 0 || width <= 0) {
    throw std::invalid_argument("image must have positive height and width");
  }

  float *host_ptr = static_cast<float *>(buf.ptr);

  SiftData sift_data;
  float *temp_memory = nullptr;
  CudaImage cuda_img;

  {
    py::gil_scoped_release release;
    InitCuda(dev_num);
    cuda_img.Allocate(width, height, iAlignUp(width, 128), false, nullptr, host_ptr);
    cuda_img.Download();

    InitSiftData(sift_data, max_pts, true, true);
    temp_memory = AllocSiftTempMemory(width, height, num_octaves, scale_up);
    ExtractSift(sift_data, cuda_img, num_octaves, init_blur, thresh, lowest_scale, scale_up, temp_memory);
    FreeSiftTempMemory(temp_memory);
  }

  const int num_pts = sift_data.numPts;
  auto keypoints = py::array_t<float>({num_pts, 2});
  auto scales = py::array_t<float>({num_pts});
  auto oris = py::array_t<float>({num_pts});
  auto scores = py::array_t<float>({num_pts});
  auto descriptors = py::array_t<float>({num_pts, 128});

  auto kp = keypoints.mutable_unchecked<2>();
  auto sc = scales.mutable_unchecked<1>();
  auto ori = oris.mutable_unchecked<1>();
  auto sco = scores.mutable_unchecked<1>();
  auto desc = descriptors.mutable_unchecked<2>();

#ifdef MANAGEDMEM
  SiftPoint *pts = sift_data.m_data;
#else
  SiftPoint *pts = sift_data.h_data;
#endif

  for (int i = 0; i < num_pts; ++i) {
    kp(i, 0) = pts[i].xpos;
    kp(i, 1) = pts[i].ypos;
    sc(i) = pts[i].scale;
    ori(i) = pts[i].orientation;
    sco(i) = pts[i].sharpness;
    for (int j = 0; j < 128; ++j) {
      desc(i, j) = pts[i].data[j];
    }
  }

  FreeSiftData(sift_data);

  return py::make_tuple(keypoints, scales, oris, scores, descriptors);
}

PYBIND11_MODULE(cudasift_py, m) {
  m.doc() = "Pybind11 wrapper for CudaSift (CPU numpy input, GPU extraction)";
  m.def(
      "extract",
      &extract_sift,
      py::arg("image"),
      py::arg("num_octaves") = 6,
      py::arg("init_blur") = 1.0f,
      py::arg("thresh") = 1.7f,
      py::arg("lowest_scale") = 0.0f,
      py::arg("scale_up") = false,
      py::arg("max_pts") = 2048,
      py::arg("dev_num") = 0,
      R"doc(
Extract SIFT features from a 2D float32 image.

Input values are used as-is (no automatic scaling). If your images are 0-1,
the default threshold from extractSift.cpp is appropriate; for 0-255 inputs,
increase the threshold accordingly.

Returns:
  keypoints (N,2), scales (N,), oris (N,), scores (N,), descriptors (N,128)
)doc");
}
