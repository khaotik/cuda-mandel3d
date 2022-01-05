cmake_minimum_required(VERSION 3.18)

project(cuda-mandel3d)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB_RECURSE CPP_SOURCE_FILES "src/*.cc" "src/*.cpp" "src/*.h")
file(GLOB_RECURSE CUDA_SOURCE_FILES "src/*.cu" "src/*.cuh")

add_executable(cudam3d ${CPP_SOURCE_FILES} ${CUDA_SOURCE_FILES})
