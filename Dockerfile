# FROM ubuntu:18.04
FROM nvidia/cuda:10.1-base-ubuntu18.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get -y update

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    net-tools \
    software-properties-common \
    swig \
    unzip \
    vim \
    wget \
    xpra \
    xserver-xorg-dev \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth=1 https://github.com/amix/vimrc.git ~/.vim_runtime
RUN sh ~/.vim_runtime/install_awesome_vimrc.sh
RUN rm -rf .vim_runtime/sources_non_forked/comfortable-motoin.vim/

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
RUN conda --version


ADD . /root/code

ENV LANG C.UTF-8

WORKDIR /root/code

RUN echo 'export PATH=/root/miniconda3/bin:$PATH' >> /root/.bashrc\
    && echo "alias traj='cd /root/code/'" >> /root/.bashrc

RUN conda update -y --name base conda && conda clean --all -y

RUN conda env create -f environment.yml

ENV PATH /opt/conda/envs/procgen/bin:$PATH
RUN /root/miniconda3/bin/conda init bash\
    && exec bash\
    && . /root/.bashrc\
    && conda activate trajectory \
    # && pip install -e .\
# RUN pip install mujoco-py==2.0.2.13
# Some conveniences
# RUN apt install -y tmux
# RUN git clone https://github.com/gpakosz/.tmux.git\
#     && ln -s -f .tmux/.tmux.conf\
#     && cp .tmux/.tmux.conf.local .