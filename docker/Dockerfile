# Define Base Image
FROM tensorflow/tensorflow:1.5.0-rc0-devel-gpu
# python 2.7, ubuntu 16.04, nvcc, cuda 9

# Install dependencies
RUN apt-get update && apt-get install -y \
  build-essential cmake unzip wget curl git vim htop \
  libopenmpi-dev zlib1g-dev python-tk \
  libsm6 libxext6 libxrender-dev libgtk2.0-dev

RUN apt-get autoclean && apt-get clean && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.cache

# Install Python dependencies
RUN pip install \
  Cython==0.29.24 \
  easydict==1.9 \
  tqdm==4.62 \
  kiwisolver==1.1.0 \
  numpy==1.16.1 \
  opencv-python==4.2.0.32 \
  Pillow==6.2.2 \
  PyYAML==5.3.1 \
  torch==1.3.0

# Install the COCO API
RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi
WORKDIR /cocoapi/PythonAPI
RUN make install

