
IMAGE_NAME="stereo_slam"

# build the container
docker build -t $IMAGE_NAME .

# start the container
docker run -it --gpus all -v .:/home/ros/ros2_ws/src/stereo_slam --network host $IMAGE_NAME bash

