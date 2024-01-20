
add_library(opencl OBJECT 
    opencl_runtime_api.cc
)

add_executable(nvopencl_wrapper
    nvopencl_wrapper.cc
)

target_link_libraries(nvopencl_wrapper OpenCL::OpenCL)