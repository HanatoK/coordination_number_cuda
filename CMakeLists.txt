cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
include(CheckIncludeFile)
project(coordination_number_cuda LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)
find_package(fmt REQUIRED)
include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(coordination_number_cuda_bin "coordination_number_cuda")
add_executable(${coordination_number_cuda_bin} common.h common.cpp gpu_kernel.h gpu_kernel.cu test.cpp)
set_property(TARGET ${coordination_number_cuda_bin} PROPERTY LANGUAGE CUDA)
target_link_libraries(${coordination_number_cuda_bin} CUDA::cudart CUDA::nvtx3 fmt)

