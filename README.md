# Modular DRL Gym Env for Robots with PyBullet

This repository includes variations from the original author integrating deep point cloud registration model. You can toggle between 1 camera and 2 cameras for real scenario experiment comparison. For point cloud registration please refer to the repository "OverlapPredator" in my repositories.

<p float="left">
  <img src="https://github.com/ignc-research/IR-DRL/blob/sim2real/docs/gifs/GifReal.gif" width="400" />
  <img src="https://github.com/ignc-research/IR-DRL/blob/sim2real/docs/gifs/GifSim.gif" width="400" /> 
</p>


## Introduction

This repository provides a platform for training virtual agents in robotics tasks using Deep Reinforcement Learning (DRL). The code is built on OpenAI Gym, Stable Baselines, and PyBullet. The system is designed to operate using a modular approach, giving the user the freedom to combine, configure, and repurpose single components like goal, robot or sensor types.

An integral part of this project is the implementation of a transition mechanism from a simulated to a real-world environment. By leveraging the functionalty of ROS (Robot Operating System) and Voxelisation techniques with Open 3D, there is a system established that can effectively deploy trained models into real-world scenarios. There they are able to deal with static and dynamic obstacles.

This project is intended to serve as a resource for researchers, robotics enthusiasts, and professional developers interested in the application of Deep Reinforcement Learning in robotics.

## Getting Started

To get started with the project, please follow the instructions in the following sections:

- [Setup](docs/SETUP.md): Instructions for setting up and installing the project.
- [Training/Evaluation](docs/TRAINING.md): Information on how to train and evaluate the models.
- [Perception](docs/Perception/Perception.md): Details about our perception Pipeline.  
- [Deployment](docs/Deployment.md): Guidelines for deploying the project in a Real World environment.

Please ensure you read through all the sections to understand how to use the project effectively.
