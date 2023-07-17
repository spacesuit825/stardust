add_executable(test_scale "testSAP.cu")

target_link_libraries(test_scale PUBLIC renderer glfw glm engine CUDA::cudart CUDA::cuda_driver)
target_include_directories(test_scale PUBLIC  renderer glfw engine)
set_target_properties(
        test_scale
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)