
file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cc) #${CMAKE_CURRENT_SOURCE_DIR}/cacti/*.cc)

add_compile_definitions(NTHREADS=4)

#list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cacti/highradix.cc)

add_library(mcpat OBJECT 
    ${SOURCES}
)

target_include_directories(mcpat PRIVATE cacti)
