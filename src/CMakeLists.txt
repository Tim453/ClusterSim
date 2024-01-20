add_subdirectory(cuda-sim)
add_subdirectory(gpgpu-sim)
add_subdirectory(intersim2)

if(GPGPUSIM_POWER_MODEL)
    add_subdirectory(accelwattch)
endif()

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

add_library(gpgpusim OBJECT 
    ${SOURCES}
)
