
set(CMAKE_CUDA_ARCHITECTURES "90;120")
set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)

add_executable(network network.cu)
add_executable(scheduler scheduler.cu)