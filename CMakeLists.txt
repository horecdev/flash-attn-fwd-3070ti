cmake_minimum_required(VERSION 3.18)

project(gemm3070 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(CUDAToolkit REQUIRED)

add_executable(attn_bench attn.cu)
target_link_libraries(attn_bench PRIVATE CUDA::cublas)