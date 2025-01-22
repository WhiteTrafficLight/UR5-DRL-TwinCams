# Modular DRL Gym Env for Robots with PyBullet (Customized Fork)

This repository is a fork of the [original IR-DRL project](https://github.com/ignc-research/IR-DRL), customized to integrate additional features and adjustments, including point cloud registration using the [OverlapPredator](https://github.com/prs-eth/OverlapPredator) model. It also includes support for toggling between one and two cameras for real-world scenario experiments.

<p float="left">
  <img src="https://github.com/ignc-research/IR-DRL/blob/sim2real/docs/gifs/GifReal.gif" width="400" />
  <img src="https://github.com/ignc-research/IR-DRL/blob/sim2real/docs/gifs/GifSim.gif" width="400" /> 
</p>

---

## Custom Modifications
1. **Training Process Visualization**:
   - Added an option in `run.py` to visualize the training process for observation.
2. **Adjusted Robot Step Size**:
   - Updated the step size of the robot end-effector in `/modular_drl_env/robot/robot.py` to improve successful training outcomes.
3. **Reward System Updates**:
   - Enhanced the reward system for better training performance.
4. **Multiple Cameras and Point Cloud Registration**:
   - Integrated variations in `IR-DRL/Sim2Real/move_DRL_main.py` to support:
     - Multiple cameras for real-world scenarios.
     - Real-time point cloud registration using the OverlapPredator model.
5. **New Training Environment**:
   - Added a custom training environment tailored for specific experiment cases.

---

## Overview
This repository provides a robust platform for training virtual agents in robotics tasks using Deep Reinforcement Learning (DRL). It supports transitioning from simulation to real-world applications with modular components like goals, robots, and sensors.

### Key Features
- **Deep Point Cloud Registration**: Integrates OverlapPredator for real-time point cloud processing.
- **Modular Environment**: Easily customizable for various tasks, including training robots to handle static and dynamic obstacles.
- **Sim-to-Real Transition**: Uses ROS and Open3D for deployment in real-world environments.

---

## Getting Started

### Installation
Follow the setup instructions from the original repository: [Setup](docs/SETUP.md).

### Key Documentation
- **Training and Evaluation**: Instructions for training and evaluating models are in [TRAINING.md](docs/TRAINING.md).
- **Perception Pipeline**: Details on perception handling can be found in [Perception.md](docs/Perception/Perception.md).
- **Deployment**: Guidelines for real-world deployment are in [Deployment.md](docs/Deployment.md).

### Additional Notes
- Ensure that `OverlapPredator` is installed and properly configured for point cloud registration. Refer to [OverlapPredator Repository](https://github.com/prs-eth/OverlapPredator).

---

## Acknowledgments
This project is based on [IR-DRL](https://github.com/ignc-research/IR-DRL). Special thanks to the original authors for their contributions.

---


