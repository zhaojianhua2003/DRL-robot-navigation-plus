FROM osrf/ros:noetic-desktop-full

ENV DEBIAN_FRONTEND=noninteractive

ENV LIBGL_ALWAYS_SOFTWARE=1
ENV GAZEBO_HEADLESS_RENDERING=1
ENV DISPLAY=
ENV GAZEBO_RESOURCE_PATH=/DRL-robot-navigation/catkin_ws/src/multi_robot_scenario/launch
ENV ROS_HOSTNAME=localhost
ENV ROS_MASTER_URI=http://localhost:11311
ENV ROS_PORT_SIM=11311

RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir torch tensorboard squaternion

RUN git clone --branch feature/docker_headless_noetic --single-branch https://github.com/reiniscimurs/DRL-robot-navigation /DRL-robot-navigation

RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    cd DRL-robot-navigation/catkin_ws && \
    catkin_make"

RUN echo "source /DRL-robot-navigation/catkin_ws/devel/setup.bash" >> /root/.bashrc

CMD ["bash"]