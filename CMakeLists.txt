cmake_minimum_required(VERSION 3.19)

project(GPGPUSim VERSION 12.0 LANGUAGES CXX C CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(BISON REQUIRED)
find_package(FLEX REQUIRED)
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)

find_package(OpenCL)
find_package(OpenMP)
find_package(OpenGL)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}") # CMake 3.9+
endif()


# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(GPGPUSIM_POWER_MODEL    			"Enable GPU Sim Power model" OFF)
option(TRACING_ON    			"Enable Tracing" OFF)
option(OPENGL_SUPPORT "Enable OpenGL Support" OFF)


set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g3")

# set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
# set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/usr/slib)
#add_compile_options(-O3 -g)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(GPGPUSIM_POWER_MODEL)
    add_compile_definitions(GPGPUSIM_POWER_MODEL)
    set(POWER_MODEL_OBJ  "$<TARGET_OBJECTS:mcpat>")
endif()

add_compile_definitions(
    # The condition for adding TRACING_ON is:
    # (Current build config is Debug) OR (CMake variable TRACING_ON is TRUE)
    $<$<OR:$<CONFIG:Debug>,$<BOOL:${TRACING_ON}>>:TRACING_ON>
)

if(OPENGL_FOUND AND OPENGL_SUPPORT)
    add_compile_definitions(OPENGL_SUPPORT)
endif()


add_compile_definitions(CUDART_VERSION=${CUDAToolkit_VERSION_MAJOR}${CUDAToolkit_VERSION_MINOR}00)
add_compile_definitions(GPU_SIM_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

include_directories(${CUDAToolkit_INCLUDE_DIRS})

add_subdirectory(cuobjdump_to_ptxplus)
add_subdirectory(libcuda)
add_subdirectory(src)

if(OpenCL_FOUND)
    add_subdirectory(libopencl)
    add_library(OpenCL SHARED
        $<TARGET_OBJECTS:opencl>
        $<TARGET_OBJECTS:gpgpusim>
        $<TARGET_OBJECTS:gpu_uarch_sim>
        $<TARGET_OBJECTS:gpgpu_ptx_sim>
        $<TARGET_OBJECTS:booksim>
        ${POWER_MODEL_OBJ}
    )
endif()

# Create a library from the source files
add_library(cudart SHARED
    $<TARGET_OBJECTS:gpgpusim>
    $<TARGET_OBJECTS:gpu_uarch_sim>
    $<TARGET_OBJECTS:cuda>
    $<TARGET_OBJECTS:gpgpu_ptx_sim>
    $<TARGET_OBJECTS:booksim>
    ${POWER_MODEL_OBJ}
)

add_library(cudart_static STATIC
    $<TARGET_OBJECTS:gpgpusim>
    $<TARGET_OBJECTS:gpu_uarch_sim>
    $<TARGET_OBJECTS:cuda>
    $<TARGET_OBJECTS:gpgpu_ptx_sim>
    $<TARGET_OBJECTS:booksim>
    ${POWER_MODEL_OBJ}
)

target_link_libraries(cudart PUBLIC ZLIB::ZLIB)
target_link_libraries(cudart_static ZLIB::ZLIB)

set_target_properties(cudart PROPERTIES
    VERSION 12
)

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/linux-so-version.txt")

# ADD_CUSTOM_TARGET(cudart12 ALL
#                   COMMAND ${CMAKE_COMMAND} -E create_symlink libcudart.so ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libcudart.so.12)
# ADD_CUSTOM_TARGET(cudart11 ALL
#                   COMMAND ${CMAKE_COMMAND} -E create_symlink libcudart.so ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libcudart.so.11.0)


add_subdirectory(benchmarks)