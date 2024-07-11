BISON_TARGET(MyParser "cuobjdump.y" "cuobjdump_parser.cc" 
    COMPILE_FLAGS "--name-prefix=cuobjdump_"    
    DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/cuobjdump_parser.h
)

FLEX_TARGET(MyLexer "cuobjdump.l" "cuobjdump_lexer.cc" COMPILE_FLAGS "--prefix=cuobjdump_")
add_flex_bison_dependency(MyLexer MyParser)

add_library(cuda OBJECT ${BISON_MyParser_OUTPUTS} ${FLEX_MyLexer_OUTPUTS} cuda_runtime_api.cc)


if(OPENGL_FOUND AND OPENGL_SUPPORT)
target_include_directories(cuda PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${CMAKE_CURRENT_BINARY_DIR}
    ${OPENGL_INCLUDE_DIR})
else()
target_include_directories(cuda PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${CMAKE_CURRENT_BINARY_DIR})
endif()

