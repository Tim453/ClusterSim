cmake_minimum_required(VERSION 3.18)

project(GPGPUSim LANGUAGES CXX C)

find_package(BISON)
find_package(FLEX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Specify the path to the source files directory
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_compile_definitions(CUDART_VERSION=11800)
# Get the list of all source files in the directory
file(GLOB SOURCES ${SOURCE_DIR}/*.cc)

include_directories(/usr/local/cuda/include)

#add_subdirectory(libcuda)
add_subdirectory(src)

# Create a library from the source files
#add_library(cudart SHARED ${SOURCES} ${BISON_MyParser_OUTPUTS})

#target_include_directories(cudart PUBLIC /usr/local/cuda/include)


#add_executable(Foo main.cpp ${BISON_MyParser_OUTPUTS})