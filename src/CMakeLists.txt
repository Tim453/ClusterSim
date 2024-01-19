add_subdirectory(cuda-sim)
add_subdirectory(gpgpu-sim)
add_subdirectory(intersim2)

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

add_library(gpgpusim OBJECT 
    ${SOURCES}
)