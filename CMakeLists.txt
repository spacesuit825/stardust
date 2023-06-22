cmake_minimum_required(VERSION 3.20)

project(
        STARDUST
        VERSION 0.0.1
        DESCRIPTION "DEM Engine"
        LANGUAGES CXX C CUDA
)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 11)
message("Processing top level CMakeLists.txt")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75) ## home
endif()

set(CMAKE_CUDA_FLAGS	"--expt-relaxed-constexpr; --generate-line-info")
find_package(OpenGL REQUIRED)
find_package(OpenMP)
find_package(CUDAToolkit REQUIRED)

#process external libraries
add_subdirectory(external)
message("External Libraries Added")

#process source files
add_subdirectory(src)

#process test scenes
add_subdirectory(scenes)