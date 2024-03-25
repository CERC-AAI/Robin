FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND="noninteractive"
ENV HOST_NAME="docker"

RUN apt-get update \
	&& apt-get install -y \
		software-properties-common \
		wget \
		git \
		git-lfs \
		unzip

# Install Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
		python3.10 \
		python3.10-dev \
		python3.10-distutils \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Pip
RUN wget https://bootstrap.pypa.io/get-pip.py \
	&& python get-pip.py \
	&& rm get-pip.py \
    && pip install --upgrade pip

# Install Robin
RUN git clone https://github.com/AGI-Collective/robin.git \
    && cd robin \
    && git checkout frontier/dev \
    && pip install -e ".[train]"

# Setup corrections
RUN pip uninstall -y bitsandbytes \
    && rm -rf /usr/local/lib/python3.10/site-packages/triton \
    && rm -rf /usr/local/lib/python3.10/site-packages/sklearn

RUN pip install flash-attn==2.3.3

RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
