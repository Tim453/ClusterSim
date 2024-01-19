file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

if(NOT GPGPUSIM_POWER_MODEL)
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/power_interface.cc)
endif()

add_library(gpu_uarch_sim OBJECT 
    ${SOURCES}
)
#target_include_directories(gpgpu_ptx_sim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
