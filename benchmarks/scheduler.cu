#include <cooperative_groups.h>
#include <cuda/std/mdspan>
#include <fstream>
#include <list>
#include "CLI11.hpp"
#include "CudaVector.cuh"

__device__ uint get_smid() {
  uint ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret));
  return ret;
}

__device__ clock_t global_time;

__device__ void d_wait() {
  auto start = clock64();

  auto now = clock64();
  for (;;) {
    now = clock64();
    auto cycles = now > start ? now - start : now + (0xffffffff - start);
    if (cycles >= 100000) {
      break;
    }
  }
  // Stored "now" in global memory here to prevent the
  // compiler from optimizing away the entire loop.
  global_time = now;
}

__global__ void busy_kernel(int *data) {
  d_wait();

  if (threadIdx.x == 0) {
    // Store the SM ID in the first element of the data array
    auto smid = get_smid();
    auto cluster = cooperative_groups::this_cluster();
    auto grid = cooperative_groups::this_grid();
    auto md_data =
        cuda::std::mdspan(data, grid.num_clusters(), cluster.num_blocks());

    md_data(grid.cluster_rank(), cluster.block_rank()) = smid;
  }
}

class DSU {
 public:
  std::vector<int> parent;
  std::vector<int> sz;  // Using 'sz' for size/rank optimization

  // Constructor: Initializes DSU for 'n' elements
  DSU(int n) {
    parent.resize(n);
    std::iota(parent.begin(), parent.end(),
              0);     // Each element is initially its own parent
    sz.assign(n, 1);  // Each set initially has size 1
  }

  // Find operation with path compression
  int find(int i) {
    if (parent[i] == i) return i;
    return parent[i] = find(parent[i]);  // Path compression
  }

  // Union operation by size/rank
  void unite(int i, int j) {
    int root_i = find(i);
    int root_j = find(j);

    if (root_i != root_j) {
      // Union by size: attach smaller tree under root of larger tree
      if (sz[root_i] < sz[root_j]) std::swap(root_i, root_j);

      parent[root_j] = root_i;
      sz[root_i] += sz[root_j];
    }
  }
};

std::vector<std::set<int>> mergeSets(
    const std::vector<std::set<int>> &listOfSets) {
  if (listOfSets.empty()) {
    return {};
  }

  // 1. Map all unique integers to contiguous DSU indices
  std::map<int, int> int_to_dsu_idx;
  int current_dsu_idx = 0;

  for (const auto &s : listOfSets) {
    for (int val : s) {
      if (int_to_dsu_idx.find(val) == int_to_dsu_idx.end()) {
        int_to_dsu_idx[val] = current_dsu_idx++;
      }
    }
  }

  // If no integers found (e.g., list of empty sets)
  if (current_dsu_idx == 0) {
    return {};
  }

  // 2. Initialize DSU structure
  DSU dsu(current_dsu_idx);

  // 3. Process each set: Union elements within the same original set
  for (const auto &s : listOfSets) {
    if (s.empty()) continue;  // Skip empty sets

    auto it = s.begin();
    int first_val_idx = int_to_dsu_idx[*it];
    ++it;

    while (it != s.end()) {
      int current_val_idx = int_to_dsu_idx[*it];
      dsu.unite(first_val_idx, current_val_idx);
      ++it;
    }
  }

  // 4. Group elements by their DSU root to form the merged sets
  std::map<int, std::set<int>> merged_sets_map;  // root_idx -> merged_set

  for (const auto &pair : int_to_dsu_idx) {
    int original_val = pair.first;
    int dsu_idx = pair.second;
    int root_idx = dsu.find(dsu_idx);
    merged_sets_map[root_idx].insert(original_val);
  }

  // Convert map values to a vector of sets
  std::vector<std::set<int>> result;
  for (const auto &pair : merged_sets_map) {
    result.push_back(pair.second);
  }

  return result;
}

// --- Helper function to print sets ---
void printSets(const std::vector<std::set<int>> &listOfSets) {
  if (listOfSets.empty()) {
    std::cout << "{}\n";
    return;
  }
  std::cout << "[\n";
  for (const auto &s : listOfSets) {
    std::cout << "  {";
    bool first = true;
    for (int val : s) {
      if (!first) {
        std::cout << ", ";
      }
      std::cout << val;
      first = false;
    }
    std::cout << "} " << s.size() << "\n";
  }
  std::cout << "]\n";
}

int main(int argc, char *argv[]) {
  int cluster_dim = 16;
  int nclusters = 30;
  int threads = 1024;

  CLI::App app{"Scheduler Microbenchmark"};

  argv = app.ensure_utf8(argv);
  app.add_option("-t", threads, "Threads per block")->default_val(threads);
  app.add_option("-n", nclusters, "Number TBCs")->default_val(nclusters);
  app.add_option("-c", cluster_dim, "Number of Blocks per cluster")
      ->default_val(cluster_dim);
  CLI11_PARSE(app, argc, argv);

  int grid_dim = cluster_dim * nclusters;

  cuda_vector<int> d_data(grid_dim);

  {
    cudaLaunchConfig_t config = {0};
    // The grid dimension is not affected by cluster launch, and is still
    // enumerated using number of blocks. The grid dimension should be a
    // multiple of cluster size.
    config.gridDim = grid_dim;
    config.blockDim = threads;

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = cluster_dim;  // Cluster size in X-dimension
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.attrs = attribute;
    config.numAttrs = 1;
    cudaFuncSetAttribute(busy_kernel,
                         cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    CUDA_CHECK(cudaLaunchKernelEx(&config, busy_kernel, d_data.data()));
  }

  auto h_data = d_data.to_host_vector();

  auto md_data = cuda::std::mdspan(h_data.data(), nclusters, cluster_dim);

  std::vector<std::set<int>> smid_list(nclusters);
  std::set<int> smids;

  for (int i = 0; i < nclusters; ++i) {
    for (int j = 0; j < cluster_dim; ++j) {
      smids.insert(md_data(i, j));
      smid_list.at(i).insert(md_data(i, j));
    }
  }
  printf("\n");

  smid_list = mergeSets(smid_list);
  printSets(smid_list);

  printf("Total unique SMIDs: %zu\n", smids.size());

  return 0;
}