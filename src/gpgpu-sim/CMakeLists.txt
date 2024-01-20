file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

if(NOT GPGPUSIM_POWER_MODEL)
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/power_interface.cc)
endif()

add_library(gpu_uarch_sim OBJECT 
    ${SOURCES}
)

if(GPGPUSIM_POWER_MODEL)
    target_include_directories(gpu_uarch_sim PRIVATE ${CMAKE_SOURCE_DIR}/src/accelwattch)
endif()