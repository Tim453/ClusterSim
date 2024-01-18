BISON_TARGET(MyParser "cuobjdump.y" "cuobjdump_parser.cc" DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/cuobjdump_parser.h)
FLEX_TARGET(MyLexer "cuobjdump.l" "cuobjdump_lexer.cc")
add_flex_bison_dependency(MyLexer MyParser)

add_library(cuda STATIC ${BISON_MyParser_OUTPUTS} ${FLEX_MyLexer_OUTPUTS} cuda_runtime_api.cc)

target_include_directories(cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
