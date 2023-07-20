file(GLOB_RECURSE ENGINE_SRC "engine/*.cpp" "engine/*.hpp" "engine/*.h" "engine/*.cuh" 
                            "engine/entities/*.cpp" "engine/entities/*.hpp" "engine/entities/*.h"
                            "engine/*.cu" "engine/cuda/*.hpp" "engine/cuda/*.cpp" "engine/cuda/*.cu" "IO/*.hpp" 
                            "IO/*.cpp" "engine/cuda/SAP/*.hpp" "engine/cuda/SAP/*.cuh" "engine/cuda/SAP/*.cu")
file(GLOB_RECURSE RENDER_SRC "render/*.cpp" "render/*.hpp" "render/*.h")

add_library(engine ${ENGINE_SRC})
target_include_directories(engine PUBLIC "./engine/" "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}" "../external/eigen-3.4.0/" OpenMP::OpenMP_CXX
           glew glfw "${PROJECT_SOURCE_DIR}/external/glm/" imgui fmt "${PROJECT_SOURCE_DIR}/external/json/" "${PROJECT_SOURCE_DIR}/external/stl_reader/" "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include/cub/")

target_link_libraries(engine PUBLIC OpenMP::OpenMP_CXX ${CUDART_LIBRARY} glew ${OPENGL_LIBRARIES} glfw imgui fmt)

get_target_property(OUT engine LINK_LIBRARIES)
message(STATUS ${OUT})

#set_source_files_properties(engine/cuda/SimulationKernel.cu PROPERTIES LANGUAGE CUDA)
#set_source_files_properties(engine/cuda/svd3.h PROPERTIES LANGUAGE CUDA)
#set_source_files_properties(engine/cuda/SimulationKernel.cuh PROPERTIES LANGUAGE CUDA)
set_source_files_properties(engine/cuda/helper_math.h PROPERTIES LANGUAGE CUDA)
#set_source_files_properties(engine/cuda/helper_matrix.h PROPERTIES LANGUAGE CUDA)
#set_source_files_properties(engine/cuda/rasterizeutils.cuh PROPERTIES LANGUAGE CUDA)
#set_source_files_properties(engine/cuda/depthpeeling.cu PROPERTIES LANGUAGE CUDA)
#set_source_files_properties(engine/cuda/rigidkernel.cu PROPERTIES LANGUAGE CUDA)
set_target_properties(
        engine
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(
    engine
    PROPERTIES 
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_library(renderer ${RENDER_SRC})

target_include_directories(renderer PUBLIC "./render/" glew glfw "${PROJECT_SOURCE_DIR}/external/glm/" fmt imgui)
target_link_libraries(renderer PUBLIC glew ${OPENGL_LIBRARIES} glfw engine fmt imgui)
