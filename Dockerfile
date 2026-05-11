FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.pixi/bin:${PATH}"
ENV CONDA_OVERRIDE_CUDA=11.8
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9"
ENV PYTHONPATH="/root/ORB-SLAM2-gsplat-pixi/gsplat:${PYTHONPATH}"

WORKDIR /root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        terminator \
        ca-certificates \
        curl \
        wget \
        git \
        vim \
        cmake \
        ninja-build \
        build-essential \
        xdg-utils \
        mesa-utils \
        libgl1 \
        libgl1-mesa-dri \
        mesa-common-dev \
        libglu1-mesa-dev \
        libnspr4 \
        libnss3 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        gedit \
        libxrender1 && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://pixi.sh/install.sh | bash

RUN git clone https://github.com/sunbei00/slam_guided_gaussian_splatting.git

WORKDIR /root/slam_guided_gaussian_splatting

RUN pixi install
RUN pixi install -e orbslam2
RUN pixi install -e gs

WORKDIR /root/slam_guided_gaussian_splatting/ORB_SLAM2
RUN pixi run -e orbslam2 bash ./build.sh

WORKDIR /root/slam_guided_gaussian_splatting

CMD ["/bin/bash"]
