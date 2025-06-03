FROM ros:foxy

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y \
    python3-colcon-common-extensions \
    ros-foxy-cv-bridge \
    build-essential \
    libgl-dev \
    libglib2.0-0 \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    numpy==1.19.5 scipy==1.10.1 opencv-python==4.8.1.78 pyyaml tqdm torch transformers einops

RUN git clone https://github.com/cvg/LightGlue.git
RUN cd LightGlue && pip3 install .

# create a workspace
RUN mkdir -p /home/ros/ros2_ws/src


SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc

# Default command
CMD ["bash"]