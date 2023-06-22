add_executable(test_scale "testDEM.cpp")

target_link_libraries(test_scale PUBLIC renderer glfw glm engine)
target_include_directories(test_scale PUBLIC  renderer glfw engine)
set_target_properties(
        test_scale
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)