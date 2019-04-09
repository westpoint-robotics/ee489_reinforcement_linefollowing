# EE489 - Robot Vision-based Path Following Using Deep Q-Network

This repository contains files related to a project involving the use of reinforcement learning to have a robot learn to follow a lined path.

The "get-hsv" camera package takes camera input and returns a 1-hot-encoded array indicating where tape is found in the picture

The "trainer" package contains a script to collect state and controller inputs used to pre-train our model

The "logger" package contains the actual model and code to run and train it (DON'T USE THIS)

The "robot_trainer" package contains the most up to date model using Keras
