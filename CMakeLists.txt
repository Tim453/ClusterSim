cmake_minimum_required(VERSION 3.19)

project(GPGPUSim VERSION 11.0 LANGUAGES CXX C CUDA)

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
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(GPGPUSIM_POWER_MODEL    			"Enable GPU Sim Power model" OFF)
option(TRACING_ON    			"Enable Tracing" OFF)
option(OPENGL_SUPPORT "Enable OpenGL Support" OFF)     


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

#set(CMAKE_CXX_FLAGS "-O3")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
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
if(TRACING_ON)
add_compile_definitions(TRACING_ON)
endif()

if(OPENGL_FOUND AND OPENGL_SUPPORT)
    add_compile_definitions(OPENGL_SUPPORT)
endif()


add_compile_definitions(CUDART_VERSION=${CUDAToolkit_VERSION_MAJOR}${CUDAToolkit_VERSION_MINOR}${CUDAToolkit_VERSION_PATCH})

include_directories(${CUDAToolkit_INCLUDE_DIRS})

add_subdirectory(cuobjdump_to_ptxplus)
add_subdirectory(libcuda)
add_subdirectory(src)
add_subdirectory(test)

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

target_link_libraries(cudart_static ZLIB::ZLIB)

set_target_properties(cudart PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

set(CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS} "-Wl,--version-script=${CMAKE_SOURCE_DIR}/linux-so-version.txt")