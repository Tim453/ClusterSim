#include <cooperative_groups.h>

#include <cuda/ptx>
#include <cuda/std/array>
#include <cuda/std/chrono>
#include <cuda/std/mdspan>
#include <format>
#include <fstream>
#include "CLI11.hpp"
#include "CudaVector.cuh"

__device__ int g_value[100];

using namespace cuda::std;

enum BECNH_CONFIG {
  LATENCY,
  BW_SEQUANTIAL,
  BW_BROADCAST,
  BW_PAIR,
  BW_RING,
  BW_ALL,
  LT_ALL
};

__device__ inline int64_t NS_Clock() {
  auto TimeSinceEpoch_ns =
      cuda::std::chrono::duration_cast<cuda::std::chrono::nanoseconds>(
          cuda::std::chrono::system_clock::now().time_since_epoch());
  return static_cast<int64_t>(TimeSinceEpoch_ns.count());
}

__device__ uint get_smid() {
  uint ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret));
  return ret;
}

template <typename T>
__device__ void init_smem(T *smem, int size, int stride = 0) {
  for (uint32_t i = 0; i < (size - stride); i++) smem[i] = (i + stride) % size;
}

template <typename T>
__device__ double measure_bw(T *mem, const int size, const int iterations,
                             const int nthreads = blockDim.x) {
  int currentIndex = threadIdx.x;

  __syncthreads();

  // Start clock
  const auto startClock = clock64();

  // Shared memory read/write loop
  for (int i = 0; i < iterations; i++) {
    currentIndex = mem[currentIndex];
  }

  __syncthreads();  // Ensure all threads finish

  // Stop clock
  const auto endClock = clock64();
  atomicAdd(&g_value[blockIdx.x], currentIndex);

  const auto duration = endClock - startClock;
  const long totalBytesTransferred = iterations * sizeof(T) * nthreads;
  const double bytesPerCycle = (double)totalBytesTransferred / duration;
  return bytesPerCycle;
}

template <typename T>
__device__ double measure_bw2(T *mem, const int size, const int iterations,
                              const int nthreads = blockDim.x) {
  int currentIndex = threadIdx.x;

  __syncthreads();

  // Start clock
  const auto startClock = clock64();

  // Shared memory read/write loop
  for (int i = 0; i < iterations; i++) {
    currentIndex = mem[currentIndex];
  }

  // Stop clock
  const auto endClock = clock64();
  atomicAdd(&g_value[blockIdx.x], currentIndex);

  const auto duration = endClock - startClock;
  const long totalBytesTransferred = iterations * sizeof(T) * nthreads;
  const double bytesPerCycle = (double)totalBytesTransferred / duration;
  return bytesPerCycle;
}

template <typename T>
__device__ double measure_latency(T *mem, int size, int iterations) {
  int my_val = 0;
  __syncthreads();
  auto start = clock64();
  for (int i = 0; i < iterations; i++) {
    my_val = mem[my_val];
  }
  const auto endClock = clock64();
  __syncthreads();
  auto result = (double)(endClock - start) / iterations;
  atomicAdd(&g_value[blockIdx.x], my_val);
  return result;
}

__global__ void network(double *latencies, int *smids, BECNH_CONFIG config,
                        const int iterations, const int stride) {
  auto cluster = cooperative_groups::this_cluster();
  auto grid = cooperative_groups::this_grid();

  auto blkid = cluster.block_rank();
  __shared__ array<long, 5000> s_data;

  if (threadIdx.x == 0) {
    auto md_smids = mdspan(smids, grid.num_clusters(), cluster.num_blocks());
    md_smids(grid.cluster_rank(), blkid) = get_smid();
    init_smem(s_data.data(), s_data.size());
  }

  cluster.sync();

  auto grid_latencies = mdspan(latencies, grid.num_clusters(),
                               cluster.num_blocks(), cluster.num_blocks());
  auto cluster_latencies =
      submdspan(grid_latencies, grid.cluster_rank(), full_extent, full_extent);

  switch (config) {
    case LATENCY:
      for (int loader = 0; loader < cluster.num_blocks(); loader++) {
        // for (int sender = 0; sender < cluster.num_blocks(); sender++) {
        if (loader == blkid)
          for (int receiver = 0; receiver < cluster.num_blocks(); receiver++) {
            // Meassure latency
            auto dsmem = cluster.map_shared_rank(s_data.data(), receiver);
            auto latency = measure_latency(dsmem, s_data.size(), iterations);
            if (threadIdx.x == 0) cluster_latencies(loader, receiver) = latency;
          }
        cluster.sync();
      }
      break;

    case BW_SEQUANTIAL: {
      for (int broadcaster = 0; broadcaster < cluster.num_blocks();
           broadcaster++) {
        for (int reader = 0; reader < cluster.num_blocks(); reader++) {
          cluster.sync();
          if (reader == blkid) {
            auto dsmem =
                broadcaster == reader
                    ? s_data.data()
                    : cluster.map_shared_rank(s_data.data(), broadcaster);
            double bw = measure_bw(dsmem, s_data.size(), iterations);
            if (threadIdx.x == 0) {
              cluster_latencies(broadcaster, reader) = bw;
            }
          }
        }
      }
    } break;

    case BW_BROADCAST: {
      for (int broadcaster = 0; broadcaster < cluster.num_blocks();
           broadcaster++) {
        auto dsmem = cluster.map_shared_rank(s_data.data(), broadcaster);
        cluster.sync();
        double bw = measure_bw(dsmem, s_data.size(), iterations);
        if (threadIdx.x == 0) {
          cluster_latencies(broadcaster, blkid) = bw;
        }
        cluster.sync();
      }
    } break;

    case BW_PAIR: {
      int target_blk =
          blkid % (2 * stride) < stride ? blkid + stride : blkid - stride;
      target_blk = target_blk % cluster.num_blocks();
      auto dsmem = cluster.map_shared_rank(s_data.data(), target_blk);
      cluster.sync();
      auto bw = measure_bw(dsmem, s_data.size(), iterations);
      if (threadIdx.x == 0) {
        cluster_latencies(blkid, target_blk) = bw;
      }
      cluster.sync();
    } break;

    case BW_RING: {
      int target_blk = (blkid + 1) % cluster.num_blocks();
      auto dsmem = cluster.map_shared_rank(s_data.data(), target_blk);
      cluster.sync();
      const auto bw = measure_bw(dsmem, s_data.size(), iterations);
      if (threadIdx.x == 0) {
        cluster_latencies(blkid, target_blk) = bw;
      }
      cluster.sync();
    } break;

    case BW_ALL: {
      const auto nthreads = blockDim.x / cluster.num_blocks();
      const int target_blk = (threadIdx.x / nthreads) % cluster.num_blocks();
      auto dsmem = cluster.map_shared_rank(s_data.data(), target_blk);
      cluster.sync();
      auto bw = measure_bw2(dsmem, s_data.size(), iterations, nthreads);
      if (threadIdx.x % nthreads == 0) {
        cluster_latencies(blkid, target_blk) = bw;
      }
      cluster.sync();

    } break;

    case LT_ALL: {
      for (int i = 0; i < cluster.num_blocks(); i++) {
        const auto nthreads = blockDim.x / cluster.num_blocks();
        const int target_blk = (threadIdx.x / nthreads) % cluster.num_blocks();
        auto dsmem = cluster.map_shared_rank(s_data.data(), target_blk);
        cluster.sync();
        if (i == blkid) {
          auto latency = measure_latency(dsmem, s_data.size(), iterations);
          if (threadIdx.x % nthreads == 0) {
            cluster_latencies(blkid, target_blk) = latency;
          }
        }
        cluster.sync();
      }
    } break;

    default:
      assert(0);
      break;
  }
  cluster.sync();
}

int main(int argc, char *argv[]) {
  CLI::App app{"NetworkBenchmark"};
  std::string filename;
  int cluster_dim;
  int threads, tbcs, iterations, stride;
  BECNH_CONFIG bench_config;
  argv = app.ensure_utf8(argv);
  app.add_option("-n", filename, "Name of the csv file")->default_val("result");
  app.add_option("-c", cluster_dim, "Blocks per Thread Block Cluster")
      ->default_val(8);
  app.add_option(
         "--config", bench_config,
         std::format("{}=Latency, {}=BW Sequential, {}=BW Broadcast, {}=BW "
                     "Pair, {}=BW Ring, {}=BW All, {}=LT All",
                     static_cast<int>(LATENCY), static_cast<int>(BW_SEQUANTIAL),
                     static_cast<int>(BW_BROADCAST), static_cast<int>(BW_PAIR),
                     static_cast<int>(BW_RING), static_cast<int>(BW_ALL),
                     static_cast<int>(LT_ALL)))
      ->default_val(LATENCY);
  app.add_option("-t", threads, "Threads per block")->default_val(1);
  app.add_option("-g", tbcs, "Number of parallel TBCs")->default_val(1);
  app.add_option("-i", iterations, "Number of iterations")->default_val(10000);
  app.add_option("-s", stride, "Stride (only for BW Pair)")->default_val(1);
  CLI11_PARSE(app, argc, argv);

  cuda_vector<double> d_latencies(cluster_dim * cluster_dim * tbcs);
  cuda_vector<int> d_smids(cluster_dim * tbcs);

  cudaFuncSetAttribute(network, cudaFuncAttributeNonPortableClusterSizeAllowed,
                       1);

  {
    cudaLaunchConfig_t config = {0};
    // The grid dimension is not affected by cluster launch, and is still
    // enumerated using number of blocks. The grid dimension should be a
    // multiple of cluster size.
    config.gridDim = cluster_dim * tbcs;
    config.blockDim = threads;

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = cluster_dim;  // Cluster size in X-dimension
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.attrs = attribute;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, network, d_latencies.data(),
                                  d_smids.data(), bench_config, iterations,
                                  stride));
  }

  auto h_latencies = d_latencies.to_host_vector();
  auto h_smids = d_smids.to_host_vector();

  auto result = mdspan(h_latencies.data(), tbcs, cluster_dim, cluster_dim);
  auto md_smids = mdspan(h_smids.data(), tbcs, cluster_dim);

  for (int cluster_rank = 0; cluster_rank < tbcs; cluster_rank++) {
    auto myfilename = std::format("{}-{}.csv", filename, cluster_rank);

    std::ofstream MyFile(myfilename);
    MyFile << "SMID,";
    for (int i = 0; i < cluster_dim; i++) {
      MyFile << md_smids(cluster_rank, i);
      if (i < cluster_dim - 1) MyFile << ",";
    }
    MyFile << "\n";
    for (int y = 0; y < cluster_dim; y++) {
      MyFile << md_smids(cluster_rank, y) << ",";
      for (int x = 0; x < cluster_dim; x++) {
        std::cout << result(cluster_rank, y, x) << " ";
        MyFile << result(cluster_rank, y, x);
        if (x < cluster_dim - 1) MyFile << ",";
      }
      std::cout << "\n";
      MyFile << "\n";
    }

    MyFile.close();
  }
}