#include <gtest/gtest.h>
#include "matmul.cuh"





TEST(MatrtixMultiply, NonCluster) {
    int MatSize = 64;
    int Tile_size = 8;
    int smem_size = Tile_size * Tile_size * sizeof(int) * 2;
    int *input_a, *input_b, *output, *gpu_output;
    input_a = new int[MatSize * MatSize];
    input_b = new int[MatSize * MatSize];
    output = new int[MatSize * MatSize];
    gpu_output = new int[MatSize * MatSize];

    init_mat(input_a, MatSize);
    init_mat(input_b, MatSize);
    matmul_cpu(input_a, input_b, output, MatSize);

    int *d_input_a, *d_input_b, *d_output;
    cudaMalloc(&d_input_a, MatSize * MatSize * sizeof(int));
    cudaMalloc(&d_input_b, MatSize * MatSize * sizeof(int));
    cudaMalloc(&d_output, MatSize * MatSize * sizeof(int));
    cudaMemcpy(d_input_a, input_a, MatSize * MatSize * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_b, input_b, MatSize * MatSize * sizeof(int),
               cudaMemcpyHostToDevice);

    dim3 num_blocks(ceil(MatSize / (float)Tile_size),
                    ceil(MatSize / (float)Tile_size), 1);

    dim3 num_threads(Tile_size, Tile_size, 1);

    matmul_gpu<<<num_blocks, num_threads, smem_size>>>(
        d_input_a, d_input_b, d_output, MatSize, Tile_size);
    cudaMemcpy(gpu_output, d_output, MatSize * MatSize * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int i = 0; i < MatSize; i++) {
        for (int j = 0; j < MatSize; j++) {
            int index = i * MatSize + j;
            EXPECT_EQ(output[index], gpu_output[index]);
        }
    }
}

TEST(MatrtixMultiply, Cluster) {
    const int MatSize = 64;
    const int Tile_size = 8;
    const int ClusterSize = 4;
    const int smem_size = Tile_size * Tile_size * sizeof(int) * 2;
    int *input_a, *input_b, *output, *gpu_output;
    input_a = new int[MatSize * MatSize];
    input_b = new int[MatSize * MatSize];
    output = new int[MatSize * MatSize];
    gpu_output = new int[MatSize * MatSize];

    init_mat(input_a, MatSize);
    init_mat(input_b, MatSize);
    matmul_cpu(input_a, input_b, output, MatSize);

    int *d_input_a, *d_input_b, *d_output;
    cudaMalloc(&d_input_a, MatSize * MatSize * sizeof(int));
    cudaMalloc(&d_input_b, MatSize * MatSize * sizeof(int));
    cudaMalloc(&d_output, MatSize * MatSize * sizeof(int));
    cudaMemcpy(d_input_a, input_a, MatSize * MatSize * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_b, input_b, MatSize * MatSize * sizeof(int),
               cudaMemcpyHostToDevice);

    dim3 num_blocks(ceil(MatSize / (float)Tile_size),
                    ceil(MatSize / (float)Tile_size), 1);

    dim3 num_threads(Tile_size, Tile_size, 1);

    // Launch the kernel
    cudaLaunchConfig_t config = {0};
    config.gridDim = num_blocks;
    config.blockDim = num_threads;
    config.dynamicSmemBytes = smem_size;

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = ClusterSize;
    attribute[0].val.clusterDim.y = ClusterSize;
    attribute[0].val.clusterDim.z = 1;
    config.attrs = attribute;
    config.numAttrs = 1;

    cudaLaunchKernelEx(&config, matmul_gpu_cluster, d_input_a, d_input_b, d_output,
                       MatSize, Tile_size);

    cudaMemcpy(gpu_output, d_output, MatSize * MatSize * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    for (int i = 0; i < MatSize; i++) {
        for (int j = 0; j < MatSize; j++) {
            int index = i * MatSize + j;
            EXPECT_EQ(output[index], gpu_output[index]);
        }
    }
}

