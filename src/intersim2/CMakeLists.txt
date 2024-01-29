BISON_TARGET(PTXParser "config.y" "y.tab.c" 
    DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/y.tab.h)
FLEX_TARGET(PTXLexer "config.l" "lex.yy.c")
add_flex_bison_dependency(PTXLexer PTXParser)

file(GLOB SOURCES *.cpp allocators/*.cpp arbiters/*cpp networks/*cpp power/*cpp routers/*cpp)

add_compile_definitions(CREATE_LIBRARY)
add_library(booksim OBJECT 
    ${SOURCES}
    ${BISON_PTXParser_OUTPUTS} 
    ${FLEX_PTXLexer_OUTPUTS}
)

target_include_directories(booksim PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/gpgpu-sim
    allocators 
    arbiters 
    networks 
    power 
    routers)
