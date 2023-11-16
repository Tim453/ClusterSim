FROM docker.io/nvidia/cuda:12.3.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y build-essential xutils-dev bison zlib1g-dev flex libglu1-mesa-dev git bash-completion \
    file libfftw3-dev
