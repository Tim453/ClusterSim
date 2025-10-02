# ClusterSim

**ClusterSim** is a cycle-accurate simulator for modeling modern GPUs executing CUDA programs.
It is built on top of [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution) and extends its functionality to support new hardware and programming features.

---

## ✨ Key Differences from GPGPU-Sim

* Support for **CUDA Thread Block Clusters** and **Distributed Shared Memory**
* Microbenchmarks (in `benchmarks/`) for studying the SM-to-SM interconnect and the Thread Block scheduler

---

## ⚙️ Setup

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

### 🔨 Build

The simulator builds as a shared library. CUDA applications linked against `cudart` (as a shared library) can be run on the simulator by adjusting the dynamic link loader path.

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

---

### ▶️ Run

By default, ClusterSim uses the configuration file at
`configs/tested-cfgs/SM90_H100/gpgpusim.config`.

You can change this via the `GPUSIM_CONFIG` environment variable.

```bash
# If not already in rpath:
source enable_simulator.sh

# Verify CUDA program links to simulator’s libcudart
ldd build/bin/network

# Run a CUDA program
./build/bin/network

# Change GPU config
export GPUSIM_CONFIG=<PATH_TO_CONFIG>/gpgpusim.config
./build/bin/network
```

---

### ⚠️ Limitations

* **Thread Block Clusters** are only supported in performance simulation mode.
* Functional-only simulation with clusters is **not supported**.
* Kernels compiled with debug symbols ("-G") do not work
---


### Notes

If you use ClusterSim in your research please cite

```bibtex
@misc{ClusterSim,
    title={ClusterSim: Modeling Thread Block Clusters in Hopper GPUs},
    url={https://hdl.handle.net/11420/57345},
    DOI={https://doi.org/10.15480/882.15858}
    author={Lühnen, Tim and Behera, Jyotirman and Tripathy, Devashree and Lal, Sohan},
    year={2025}
}
```

Also cite the original [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution) project.

