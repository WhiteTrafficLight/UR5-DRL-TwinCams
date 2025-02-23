# UR5-DRL-TwinCams

This repository is based on [IR-DRL](https://github.com/ignc-research/IR-DRL) with minor modifications to integrate **dual RGB-D cameras** for improved path-planning using Deep Reinforcement Learning (DRL). The occlusion issue in single-camera setups is mitigated by adding a second **ASUS Xtion Pro** camera and aligning point clouds using **PREDATOR** (Point Cloud Registration model).

This repository was used in my **master's thesis**:
> **"Deep Reinforcement Learning and Point Cloud Registration for Collision Avoidance and Path-Planning in Industrial Robotics"**  
> 📄 **[Master's Thesis Link](https://drive.google.com/file/d/1zJlFOs_0xcjcc7gajkvttXdFDhaQVWaQ/view)** 
> 📄 **[Medium Blog Summary](https://medium.com/@jhyu7703/deep-reinforcement-learning-and-point-cloud-registration-for-collision-avoidance-and-path-planning-92d8562158df)**

## 🏗️ Pipeline Overview
![Pipeline Overview](path/to/pipeline_image.png)  
*(Replace with actual image path)*  

The pipeline consists of:
- **DRL-based path-planning for UR5**  
- **Voxel-based obstacle representation**  
- **Point cloud fusion using PREDATOR**  
- **Simulation in PyBullet & real-world execution via ROS1 Noetic**  

## 📹 Demo Video
▶️ **[YouTube Video](https://www.youtube.com/watch?v=y-R9BKT0rpw)** 

## 🔑 Key Modifications
- **Dual-camera integration** to resolve occlusion in single-camera setups check OverlapPredator/ros_nodes directory.
- **IK-solver-based agent design fine-tuning:** In the original setup, the agent’s **Δx, Δy, Δz** range was too small, leading the agent to optimize for reducing the shaking penalty rather than reaching the target. Expanding this range resolved the issue.
- **Sim2Real/move_DRL_main.py** adapted for dual-camera experiments.
- **BiRRT-based dynamic sampling path-planning attempt**, but proved ineffective.

> **Note:** The core DRL training framework remains **mostly unchanged** from the original IR-DRL repository. The primary modifications lie in **data preprocessing (point cloud alignment)** and **integration of OverlapPredator** for better real-world adaptability. Reference to OverlapPredator/scripts/experimental for the experimental code

## 🚀 Installation
This project runs on **ROS1 Noetic** with the following dependencies:

1. Install **ROS1 Noetic** (Ubuntu 20.04)  
   ```bash
   sudo apt install ros-noetic-desktop-full
   ```
2. Clone the repository:  
   ```bash
   git clone https://github.com/WhiteTrafficLight/UR5-DRL-TwinCams.git
   cd UR5-DRL-TwinCams
   ```
3. Install dependencies via Conda:  
   ```bash
   conda env create -f DRL.yml
   conda activate DRL
   ```
4. Install additional ROS packages:  
   ```bash
   sudo apt install ros-noetic-openni2-camera ros-noetic-universal-robot
   ```
5. Ensure you have **two ASUS Xtion Pro cameras** connected.

## 🎯 Usage
### Dual-Camera Mode
1. **Start ROS core**  
   ```bash
   roscore
   ```
2. **Launch UR5 Robot Drivers**  
   ```bash
   roslaunch ur_robot_driver ur5_bringup.launch
   ```
3. **Launch OpenNI2 Camera Drivers for both cameras
   ```bash
   roslaunch openni2_launch openni2.launch camera:=camera
   roslaunch openni2_launch openni2.launch camera:=camera2
   ```      
4. **Run Point Cloud Matcher**  
   ```bash
   python OverlapPredator/ros_nodes/ros_pointcloud_matcher_static.py
   ```
5. **Run DRL-based Path-Planning**  
   ```bash
   python Sim2Real/move_DRL_main.py
   ```
   
## 📜 Acknowledgments
This repository is based on [IR-DRL](https://github.com/ignc-research/IR-DRL).  
The PREDATOR model used for point cloud registration comes from [OverlapPredator](https://github.com/prs-eth/OverlapPredator.git).







