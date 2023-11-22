#define _CG_HAS_CLUSTER_GROUP
#include <cooperative_groups.h>

// Distributed Shared memory histogram kernel
__global__ void clusterHistkernel(int *bins, const int nbins,
                                  const int bins_per_block,
                                  const int *__restrict__ input,
                                  size_t array_size) {
    extern __shared__ int smem[];
    namespace cg = cooperative_groups;
    int tid = cg::this_grid().thread_rank();

    // Cluster initialization, size and calculating local bin offsets.
    cg::cluster_group cluster = cg::this_cluster();
    int cluster_size = cluster.dim_blocks().x;

    for (int k = 0; k < nbins; k += bins_per_block * cluster_size) {
        for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x) {
            smem[i] = 0;  // Initialize shared memory histogram to zeros
        }
        cluster.sync();

        for (int i = tid; i < array_size; i += blockDim.x * gridDim.x) {
            int ldata = input[i] - k;
            if (ldata >= 0 && ldata < bins_per_block * cluster_size) {
                // Find destination block rank and offset for computing
                // distributed shared memory histogram
                int dst_block_rank = (int)(ldata / bins_per_block);
                int dst_offset = ldata % bins_per_block;

                // Pointer to target block shared memory
                int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);

                // Perform atomic update of the histogram bin
                atomicAdd(dst_smem + dst_offset, 1);
            }
        }
        cluster.sync();
        int *lbins = bins + cluster.block_rank() * bins_per_block + k;
        for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x) {
            if (i + k < nbins) atomicAdd(&lbins[i], smem[i]);
        }
        cluster.sync();
    }
}



__global__ void Hist_kernel(int *bins, const int nbins,
                            const int bins_per_block,
                            const int *__restrict__ input, size_t array_size) {
    extern __shared__ int smem[];
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;

    for (int k = 0; k < nbins; k += bins_per_block) {
        for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x) {
            smem[i] = 0;  // Initialize shared memory histogram to zeros
        }
        __syncthreads();

        for (int i = tid; i < array_size; i += blockDim.x * gridDim.x) {
            // Perform atomic update of the histogram bin
            int array_value = input[i] - k;
            if (array_value >= 0 && array_value < bins_per_block)
                atomicAdd(&smem[array_value], 1);
        }
        __syncthreads();

        // Perform global memory histogram, using the local distributed memory
        // histogram
        for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x) {
            if (i + k < nbins) {
                atomicAdd(&bins[i + k], smem[i]);
            }
        }
        __syncthreads();
    }
}

void init_array(int *array, int array_size, int nbins) {
    for (size_t i = 0; i < array_size; i++) {
        array[i] = rand() % nbins;
    }
}

void histogram_cpu(const int *input, int size, int nbins, int *bins) {
    for (size_t i = 0; i < nbins; i++) {
        bins[i] = 0;
    }
    for (size_t i = 0; i < size; i++) {
        bins[input[i]]++;
    }
}

