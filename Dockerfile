FROM docker.io/nvidia/cuda:12.3.0-devel-ubuntu22.04

ENV CUDA_INSTALL_PATH=/usr/local/cuda
ENV PATH=$PATH:/usr/local/cuda/

RUN apt-get update && apt-get install -y build-essential xutils-dev bison zlib1g-dev flex libglu1-mesa-dev git bash-completion \
    file libfftw3-dev libgtest-dev
