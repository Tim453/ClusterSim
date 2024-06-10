BISON_TARGET(PTXParser "ptx.y" "ptx.tab.cc"
    COMPILE_FLAGS "--name-prefix=ptx_"
    DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/ptx.tab.h)
FLEX_TARGET(PTXLexer "ptx.l" "lex.ptx_.cc")
add_flex_bison_dependency(PTXLexer PTXParser)

BISON_TARGET(ELFParser "elf.y" "elf_parser.cc" 
    COMPILE_FLAGS "--name-prefix=elf_"
    DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/elf_parser.hh)
FLEX_TARGET(ELFLexer "elf.l" "elf_lexer.cc" COMPILE_FLAGS "--prefix=elf_")
add_flex_bison_dependency(ELFLexer ELFParser)

BISON_TARGET(HEADERParser "header.y" "header_parser.cc" 
    COMPILE_FLAGS "--name-prefix=header_"
    DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/header_parser.hh)
FLEX_TARGET(HEADERLexer "header.l" "header_lexer.cc" COMPILE_FLAGS "--prefix=header_")
add_flex_bison_dependency(HEADERLexer HEADERParser)

BISON_TARGET(SASSParser "sass.y" "sass_parser.cc"
    COMPILE_FLAGS "--name-prefix=sass_"
    DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/sass_parser.hh)
FLEX_TARGET(SASSLexer "sass.l" "sass_lexer.cc"  COMPILE_FLAGS "--prefix=sass_")
add_flex_bison_dependency(SASSLexer SASSParser)

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

add_executable(cuobjdump_to_ptxplus 
    ${SOURCES}
    ${BISON_PTXParser_OUTPUTS} 
    ${FLEX_PTXLexer_OUTPUTS}
    ${BISON_ELFParser_OUTPUTS} 
    ${FLEX_ELFLexer_OUTPUTS}
    ${BISON_HEADERParser_OUTPUTS} 
    ${FLEX_HEADERLexer_OUTPUTS}
    ${BISON_SASSParser_OUTPUTS} 
    ${FLEX_SASSLexer_OUTPUTS}
)

target_include_directories(cuobjdump_to_ptxplus PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda-sim 
    ${CMAKE_CURRENT_BINARY_DIR})
