

# DRL-Nav-Algorithms 

这是一个基于 **ROS Noetic** 和 **PyTorch** 的移动机器人深度强化学习 (DRL) 导航框架。

本项目基于 [DRL-robot-navigation](https://github.com/reiniscimurs/DRL-robot-navigation) 进行二次开发。原项目主要使用 TD3 算法在 Gazebo 仿真环境中控制移动机器人避障并导航至目标点。本项目在保留原有环境交互逻辑（激光雷达感知 + 极坐标目标点）的基础上，对代码进行了重构与扩展，增加了多种DRL算法。

<p align="center">
<img width=100% src="training.gif" alt="Training Example">
</p>

## ✨ 主要特性 (Key Features)

相较于原版，本项目主要包含以下改进：

1. **多算法支持 (Multi-Algorithm Support)**：
* 不仅仅局限于原有的 **TD3** (Twin Delayed DDPG)。
* 新增 **On-Policy** 算法：
    * **PPO** (Proximal Policy Optimization)
    * **VPG** (Vanilla Policy Gradient)
    * **TRPO**
* 新增 **Off-Policy** 算法：
    ***SAC** (Soft Actor-Critic)。


2. **增强可视化 (Enhanced Visualization)**：
* 增加了训练过程中的详细日志记录。
* 集成 `tqdm` 进度条，实时监控训练进度。


3. **模块化架构 (Modular Code)**：
* 重构了 `Agent` 类与训练逻辑。
* 新增 `RL_utils` 文件，将算法核心与训练解耦，使得添加新算法或修改网络结构更加容易。
* 增加环境接口，便于用户适配自定义的 Gazebo 环境。


4. **环境适配**：
* 默认适配模拟的 [3D Velodyne 传感器](https://github.com/lmark1/velodyne_simulator)，同时也兼容标准 2D 激光雷达。



## 🛠️ 系统要求 (Prerequisites)

* **操作系统**: Ubuntu 20.04
* **ROS 版本**: [ROS Noetic](http://wiki.ros.org/noetic/Installation) 
* **显卡驱动**: 建议安装 CUDA 以支持 PyTorch GPU 加速训练（CPU也可以）

## ⚙️ 环境配置与安装 (Environment Setup)

### 1. 安装 ROS Noetic

如果尚未安装 ROS Noetic，请参考 [官方指南](http://wiki.ros.org/noetic/Installation/Ubuntu) 进行安装。

### 2. 配置 Conda 环境

为了避免与系统 Python 环境冲突，建议使用 Conda 管理依赖。由于 ROS Noetic 默认使用 Python 3.8，我们需要创建一个 Python 3.8 的虚拟环境。

```shell
# 1. 创建名为 drl-nav 的环境，指定 python 版本为 3.8
conda create -n drl-nav python=3.8

# 2. 激活环境
conda activate drl-nav

# 3. 安装 PyTorch (根据你的 CUDA 版本选择，这里以 CUDA 11.8 为例)
# 如果没有 GPU，请去掉 --index-url 部分，直接 pip install torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. 安装其他必要依赖
# rospkg 和 catkin_pkg 是必须的，用于在 Conda 环境中调用 ROS 接口
pip install rospkg catkin_pkg tensorboard matplotlib numpy tqdm

```

### 3. 克隆仓库

```shell
cd ~
git clone https://github.com/zhaojianhua2003/DRL-Nav-Algorithms

```

### 4. 编译 ROS 工作空间

注意：编译通常建议在系统环境下进行，但在运行 Python 训练脚本时需要激活 Conda 环境。

```shell
cd ~/DRL-Nav-Algorithms/catkin_ws

# 编译工作空间
catkin_make_isolated

# 刷新环境配置
source devel_isolated/setup.bash

```

## 🏃‍♂️ 运行与训练 (Usage)



### 终端准备

打开一个新的终端：

```shell
# 1. Source ROS 和工作空间配置
export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311
export ROS_PORT_SIM=11311
export GAZEBO_RESOURCE_PATH=~/DRL-Nav-Algorithms/catkin_ws/src/multi_robot_scenario/launch
cd ~/DRL-Nav-Algorithms/catkin_ws
source devel_isolated/setup.bash

# 2. 激活 Python 环境
conda activate drl-nav



```

### 启动训练

本项目支持多种算法，根据命名运行训练脚本。以 TD3 为例：

```shell
cd ~/DRL-Nav-Algorithms/DRL-algorithms
# 运行带有 Velodyne 雷达的 TD3 训练脚本
python3 train_TD3_BaseWorld.py

```


### 监控训练进度

使用 Tensorboard 查看损失函数曲线和奖励变化：

```shell
cd ~/DRL-Nav-Algorithms/DRL-algorithms  
tensorboard --logdir runs

```

### 终止训练

如果需要强制结束所有 ROS 节点和训练进程：

```shell
killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient python python3

```

### 测试模型

训练完成后，加载保存的模型进行测试：

```shell
cd ~/DRL-Nav-Algorithms/DRL-algorithms
python3 test_TD3_BaseWorld.py

```

## 📷 仿真环境概览

**Gazebo 环境:**

<p align="center">
<img width=80% src="BaseWorld.png">
</p>

**Rviz 传感器视图:**

<p align="center">
<img width=80% src="velodyne.png">
</p>

## 🔗 致谢 (Acknowledgments)

本项目主要参考并基于以下仓库开发，感谢原作者的贡献：

* [DRL-robot-navigation](https://github.com/reiniscimurs/DRL-robot-navigation) by reiniscimurs

---

**Developers**: Zhao Jianhua (赵剑华)