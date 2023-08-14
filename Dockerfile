FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

# Install Python 3.8.10
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y install git make cmake build-essential libboost-all-dev python3-pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN pip install --upgrade pip

# Install PIP requirements
ENV PIP_ROOT_USER_ACTION=ignore
RUN python -m pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
COPY requirements.txt /root
RUN cd /root && pip install -r requirements.txt

# Copy & Install Python-project
COPY cvat2trainedyolo /root/cvat2trainedyolo
COPY *.py /root/
COPY .git /root/.git
COPY yolov5 /root/yolov5
WORKDIR /root

# CMD /bin/bash
