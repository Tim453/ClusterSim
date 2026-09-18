# ClusterSim

**ClusterSim** is a cycle-accurate simulator for modeling modern GPUs executing CUDA programs.
It is built on top of [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution) and extends its functionality to support new hardware and programming features.

---

## New features compared to GPGPU-Sim 4.0

* Group cores into GPCs with `gpgpu_n_cores_per_gpc` parameter
* New Special registers: `%cluster_ctaid`, `%cluster_ctarank`, `%cluster_nctaid`, `%cluster_nctarank`...
* New Instructions:  `mapa`, `barrier.cluster`
* New directives: `.explicitcluster`, `.maxclusterrank`, `.reqnctapercluster`
* New API calls: `cudaLaunchKernelExC`, `cudaMallocManaged`
* Simulation model for the SM-to-SM interconnect

---

## Setup

### Dependencies

Install required packages:

```bash
sudo apt-get install -y build-essential xutils-dev bison zlib1g-dev flex \
    libglu1-mesa-dev git cmake ninja-build clang clang-format
```

Make sure the CUDA toolkit is in your `PATH`:

```bash
export PATH=$PATH:/usr/local/cuda/bin
```

---

### Build

The simulator builds as a shared library. CUDA applications linked against `cudart` (as a shared library) can be run on the simulator by adjusting the dynamic link loader path.

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

---

### Run

By default, ClusterSim uses the configuration file at
`configs/tested-cfgs/SM90_H100/gpgpusim.config`.

You can change this via the `GPUSIM_CONFIG` environment variable.

```bash
# If not already in rpath
source enable_simulator.sh

# Verify the CUDA program links to the simulator's libcudart
ldd build/bin/network

# Run a CUDA program
./build/bin/network

# Change the GPU configuration
export GPUSIM_CONFIG=<PATH_TO_CONFIG>/gpgpusim.config
./build/bin/network
```

---


### Notes

If you use ClusterSim in your research please cite

```bibtex
@INPROCEEDINGS {11241997,
  author = { Luhnen, Tim and Behera, Jyotirman and Tripathy, Devashree and Lal, Sohan },
  booktitle = { 2025 IEEE International Symposium on Workload Characterization (IISWC) },
  title = {{ ClusterSim: Modeling Thread Block Clusters in Hopper GPUs }},
  year = {2025},
  pages = {504-515},
  doi = {10.1109/IISWC66894.2025.00048},
  url = {https://doi.ieeecomputersociety.org/10.1109/IISWC66894.2025.00048},
  publisher = {IEEE Computer Society},
  address = {Los Alamitos, CA, USA},
  month =Oct
}

@inproceedings{sachs2026accelerating,
  title={Accelerating GPGPU Simulation by Strategically Parallelizing the Compute Bottleneck},
  doi={10.4230/OASIcs.PARMA-DITAM.2026.6},
  url={https://drops.dagstuhl.de/entities/document/10.4230/OASIcs.PARMA-DITAM.2026.6}
  author={Sachs, Jakob and Lühnen, Tim and Lal, Sohan},
  booktitle={17th Workshop on Parallel Programming and Run-Time Management Techniques for Many-Core Architectures and 15th Workshop on Design Tools and Architectures for Multicore Embedded Computing Platforms (PARMA-DITAM 2026)},
  year={2026},
  organization={Schloss Dagstuhl--Leibniz-Zentrum für Informatik}
}
```

Also cite the original [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution) project.

