#define _CG_HAS_CLUSTER_GROUP
#include <cooperative_groups.h>


__global__ void matmul_gpu_cluster(const int *d_input_a, const int *d_input_b,
                               int *d_output, int matsize, int tile_size) {
    cooperative_groups::cluster_group cluster =
        cooperative_groups::this_cluster();

    int cluster_size = cluster.dim_blocks().x;

    extern __shared__ int smem_a[];
    int *smem_b = &smem_a[tile_size * tile_size];
    
    // Shared memory region to store the remote content
    int *r_smem_a = &smem_b[tile_size * tile_size];
    int *r_smem_b = &r_smem_a[tile_size * tile_size];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int Pvalue = 0;
    dim3 block_index = cluster.block_index();

    for (int ph = 0; ph < ceil(matsize / (float)tile_size);
         ph += cluster_size) {
        int a_x_offset = ((block_index.x + ph) * tile_size + tx);
        int b_y_offset = ((block_index.y + ph) * tile_size + ty);

        // Load tile of Matrix A into shared memory
        if (a_x_offset < matsize) {
            smem_a[ty * tile_size + tx] = d_input_a[y * matsize + a_x_offset];
        } else {
            smem_a[ty * tile_size + tx] = 0;
        }
        // Load tile of Matrix B into shared memory
        if (b_y_offset < matsize) {
            smem_b[ty * tile_size + tx] = d_input_b[b_y_offset * matsize + x];
        } else {
            smem_b[ty * tile_size + tx] = 0;
        }

        cluster.sync();

        if (x < matsize && y < matsize) {
            for (int i = 0; i < cluster_size; i++) {
                int rank_a = i + block_index.y * cluster_size;
                int rank_b = i * cluster_size + block_index.x;

                int *smem_ptr_a = cluster.map_shared_rank(smem_a, rank_a);
                int *smem_ptr_b = cluster.map_shared_rank(smem_b, rank_b);

                // Copy remote smem to local smem
                r_smem_a[ty * blockDim.x + tx] = smem_ptr_a[ty * blockDim.x + tx];
                r_smem_b[ty * blockDim.x + tx] = smem_ptr_b[ty * blockDim.x + tx];
                __syncthreads();
                for (int k = 0; k < tile_size; ++k) {
                    Pvalue += r_smem_a[ty * tile_size + k] *
                              r_smem_b[k * tile_size + tx];
                }
            }
        }
        cluster.sync();
    }
    if (x < matsize && y < matsize) {
        d_output[y * matsize + x] = Pvalue;
    }
}

__global__ void matmul_gpu(const int *d_input_a,
                                       const int *d_input_b, int *d_output,
                                       int matsize, int tile_size) {
    extern __shared__ int smem_a[];
    int *smem_b = &smem_a[tile_size * tile_size];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int Pvalue = 0;

    for (int ph = 0; ph < ceil(matsize / (float)tile_size); ph += 1) {
        int a_x_offset = ph * tile_size + tx;
        int b_y_offset = ph * tile_size + ty;

        // Load tile of Matrix A into shared memory
        if (a_x_offset < matsize) {
            smem_a[ty * tile_size + tx] = d_input_a[y * matsize + a_x_offset];
        } else {
            smem_a[ty * tile_size + tx] = 0;
        }
        // Load tile of Matrix B into shared memory
        if (b_y_offset < matsize) {
            smem_b[ty * tile_size + tx] = d_input_b[b_y_offset * matsize + x];
        } else {
            smem_b[ty * tile_size + tx] = 0;
        }

        __syncthreads();

        if (x < matsize && y < matsize) {
            for (int k = 0; k < tile_size; ++k) {
                Pvalue +=
                    smem_a[ty * tile_size + k] * smem_b[k * tile_size + tx];
            }
        }
        __syncthreads();
    }
    if (x < matsize && y < matsize) {
        d_output[y * matsize + x] = Pvalue;
    }
}


void matmul_cpu(const int *input_a, const int *input_b, int *output, int size) {
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int pval = 0;
            for (int k = 0; k < size; k++) {
                pval += input_a[y * size + k] * input_b[k * size + x];
            }
            output[y * size + x] = pval;
        }
    }
}

void init_mat(int *input, int size) {
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            input[y * size + x] = rand() % 10;
        }
    }
}



void cmp_mat(int *input1, int *input2, int size) {
    unsigned errors = 0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int index = i * size + j;
            if (input1[index] != input2[index]) {
                errors++;
            }
        }
    }
}