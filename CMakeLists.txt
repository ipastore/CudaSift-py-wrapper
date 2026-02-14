cmake_minimum_required(VERSION 2.6)

project(cudaSift)
set(cudaSift_VERSION_MAJOR 2)
set(cudaSift_VERSION_MINOR 0)
set(cudaSift_VERSION_PATCH 0)

set(CPACK_PACKAGE_VERSION_MAJOR "${cudaSift_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${cudaSift_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${cudaSift_VERSION_PATCH}")
set(CPACK_GENERATOR "ZIP")
include(CPack)

find_package(OpenCV REQUIRED)
find_package(CUDA)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)
find_package(pybind11 CONFIG REQUIRED)

if (CMAKE_CUDA_ARCHITECTURES)
  set(CUDASIFT_CUDA_ARCHS ${CMAKE_CUDA_ARCHITECTURES})
else()
  set(CUDASIFT_CUDA_ARCHS 75)
endif()
set(CUDASIFT_ARCH_FLAGS "")
foreach(arch ${CUDASIFT_CUDA_ARCHS})
  list(APPEND CUDASIFT_ARCH_FLAGS "-gencode=arch=compute_${arch},code=sm_${arch}")
endforeach()

if (WIN32)
  set(EXTRA_CXX_FLAGS "/DVERBOSE /D_CRT_SECURE_NO_WARNINGS ")
  list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-O2;-DVERBOSE")
endif()
if (UNIX)
  if (APPLE)
    set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2")
    list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-O2;-DVERBOSE")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -msse2 ")
    list(APPEND CUDA_NVCC_FLAGS "-lineinfo;--compiler-options;-O2;-D_FORCE_INLINES;-DVERBOSE_NOT")
  endif()
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fPIC")
endif()
list(APPEND CUDA_NVCC_FLAGS ${CUDASIFT_ARCH_FLAGS})

set(cuda_sources
  cudaImage.cu  
  cudaImage.h  
  cudaSiftH.cu 
  cudaSiftH.h  
  matching.cu  
  cudaSiftD.h  
  cudaSift.h  
  cudautils.h
)  

set(sources
  geomFuncs.cpp  
  mainSift.cpp
)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR} 
)

#SET(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(cudasift ${cuda_sources} ${sources} OPTIONS ${CUDASIFT_ARCH_FLAGS})

#cuda_add_executable(l2net l2netD.cu OPTIONS -arch=sm_75)

set_target_properties(cudasift PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"			   
)

target_link_libraries(cudasift ${CUDA_cudadevrt_LIBRARY} ${OpenCV_LIBS})
#  /usr/local/cuda/lib64/libcudadevrt.a ${OpenCV_LIBS} 
#)

if (pybind11_FOUND)
  cuda_add_library(cudasift_cuda STATIC ${cuda_sources} OPTIONS ${CUDASIFT_ARCH_FLAGS})
  set_target_properties(cudasift_cuda PROPERTIES
    COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"
    POSITION_INDEPENDENT_CODE ON
  )
  pybind11_add_module(cudasift_py python/cudasift_pybind.cpp)
  target_include_directories(cudasift_py PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(cudasift_py PRIVATE cudasift_cuda ${CUDA_cudadevrt_LIBRARY} ${OpenCV_LIBS})
endif()
 
install(FILES 
  ${cuda_sources} 
  ${sources}
  cudaSiftD.cu
  CMakeLists.txt
  Copyright.txt
  DESTINATION .
)
install(FILES data/left.pgm data/righ.pgm
  DESTINATION data
)
