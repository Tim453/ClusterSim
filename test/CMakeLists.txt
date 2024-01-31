find_package(GTest)

remove_definitions(CUDART_VERSION)

set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)
include(GoogleTest)
add_executable(test_kernel test.cu)
set_target_properties(test_kernel PROPERTIES CUDA_ARCHITECTURES "90")
target_link_libraries(test_kernel GTest::gtest_main)

gtest_discover_tests(test_kernel 
    PROPERTIES 
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/"
        ENVIRONMENT "GPUSIM_CONFIG=/home/tim/tbc_bench/gpgpu-sim_distribution/test/test.config"
        ENVIRONMENT "PTX_SIM_USE_PTX_FILE=1.ptx"
        ENVIRONMENT "PTX_SIM_KERNELFILE=_1.ptx"
        ENVIRONMENT "CUOBJDUMP_SIM_FILE=jj"
)
