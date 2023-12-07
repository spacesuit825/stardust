file(GLOB_RECURSE ENGINE_SRC 
                            "engine/*.cu"
                            "engine/*/*.cu"
                            "engine/*.cuh"
                            "engine/*.cpp"
                            "engine/*.hpp"
                            "engine/*/*.hpp"
                            "engine/*.c"
                            "engine/*.h"
)


add_library(engine ${ENGINE_SRC})

target_include_directories(engine PUBLIC ENGINE_SRC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(engine PUBLIC ENGINE_SRC OpenMP::OpenMP_CXX ${CUDART_LIBRARY} glew ${OPENGL_LIBRARIES} glfw imgui fmt)

set_source_files_properties(engine/stardustCollision/stardustBroadPhase/stardustLBVH.cu PROPERTIES LANGUAGE CUDA)
set_source_files_properties(engine/stardustCollision/stardustNarrowPhase/stardustMPR.cu PROPERTIES LANGUAGE CUDA)

add_executable(test_scale "engine/testEngine.cpp")

set(CMAKE_BUILD_TYPE Debug)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")



target_link_libraries(test_scale PUBLIC engine CUDA::cudart CUDA::cuda_driver)
target_include_directories(test_scale PUBLIC engine)
set_target_properties(
        test_scale
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)