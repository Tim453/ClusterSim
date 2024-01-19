cmake_minimum_required(VERSION 3.18)

project(GPGPUSim VERSION 11.0 LANGUAGES CXX C)

find_package(BISON)
find_package(FLEX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(GPGPUSIM_POWER_MODEL    			"Enable GPU Sim Power model" OFF)
option(TRACING_ON    			"Enable Tracing" OFF)

# Specify the path to the source files directory
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(GPGPUSIM_POWER_MODEL)
add_compile_definitions(GPGPUSIM_POWER_MODEL)
endif()
if(TRACING_ON)
add_compile_definitions(TRACING_ON)
endif()

add_compile_definitions(CUDART_VERSION=11800)

include_directories(/usr/local/cuda/include)

add_subdirectory(libcuda)
add_subdirectory(src)

# Create a library from the source files
add_library(cudart SHARED  
    $<TARGET_OBJECTS:gpgpusim>
    $<TARGET_OBJECTS:gpu_uarch_sim> 
#    $<TARGET_OBJECTS:cuda>
    $<TARGET_OBJECTS:gpgpu_ptx_sim> 
    $<TARGET_OBJECTS:booksim>
)

set_target_properties(cudart PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

set(CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS} "-Wl,--version-script=${CMAKE_SOURCE_DIR}/linux-so-version.txt")