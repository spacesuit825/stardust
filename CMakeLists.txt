cmake_minimum_required(VERSION 3.20)

project(
        STARDUST
        VERSION 0.0.1
        DESCRIPTION "DEM Engine"
        LANGUAGES CXX C CUDA
)

enable_testing()

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
message("Processing top level CMakeLists.txt")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "35;50;52;60;61;70;75;80;86")
endif()

set(CMAKE_CUDA_FLAGS	"--expt-relaxed-constexpr; --generate-line-info")
find_package(OpenGL REQUIRED)
find_package(OpenMP)
find_package(CUDAToolkit REQUIRED)
find_package(Thrust REQUIRED)
find_package(cub REQUIRED)

#process external libraries
#add_subdirectory(external)
message("External Libraries Added")

#process source files
#add_subdirectory(src)

#process test scenes
#add_subdirectory(scenes)
add_subdirectory(sandbox)