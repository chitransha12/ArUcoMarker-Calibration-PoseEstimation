Aruco Marker And Calibration And Pose Estimation

Overview
This project provides a complete workflow for camera calibration and pose estimation using Aruco markers. It allows you to capture images, calibrate your camera to remove lens distortion, and estimate the 3D pose of markers in real-time. Ideal for robotics, computer vision, and AR applications.

Features
Live camera preview during calibration
Automatic generation of camera matrix and distortion coefficients
Pose estimation of Aruco markers in real-time
Visualization of marker axes and IDs on the image
Save and load calibration parameters for future use

Requirements
Python 3.x
OpenCV (opencv-python, opencv-contrib-python)
NumPy
Glob
OS (for file handling)

Install dependencies with:
pip install opencv-python opencv-contrib-python numpy

Usage
1. Camera Calibration
Place Aruco markers in front of the camera.
Capture multiple images to cover different angles.
Run the calibration script to generate cameraMatrix and distCoeffs.

2. Pose Estimation
Load calibration parameters.
Detect Aruco markers in the camera feed.
Estimate pose and visualize axes.

# Example
import cv2
import cv2.aruco as aruco
import numpy as np

# Load camera parameters
camera_matrix = np.loadtxt("cameraMatrix.txt")
dist_coeffs = np.loadtxt("distCoeffs.txt")

# Detect Aruco markers
# Estimate pose
# Draw axis on markers



Notes
Marker size (in cm) should be correctly defined for accurate pose estimation.
Ensure enough varied images during calibration for better results.
Works for both live camera feed and static images.
