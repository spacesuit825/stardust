add_executable(test_scale "testGJK.cu" "util.hpp")

set(CMAKE_BUILD_TYPE Debug)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")



target_link_libraries(test_scale PUBLIC CUDA::cudart CUDA::cuda_driver)
#target_include_directories(test_scale PUBLIC)
set_target_properties(
        test_scale
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)