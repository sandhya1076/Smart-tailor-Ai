# Smart-Tailor-AI

## AI-Based Human Body Measurement and Clothing Size Recommendation System Using YOLO Pose

Smart-Tailor-AI is an AI-based system designed to automate human body measurement from images and provide an experimental clothing size recommendation.

## Problem Statement

Manual body measurement for tailoring is time-consuming and requires assistance. Online clothing selection can also be difficult when users do not know their accurate body measurements or suitable clothing size.

This project aims to develop an AI-based solution that extracts approximate body measurements from a user's image and provides a clothing size recommendation using computer vision and deep learning.

## Objectives

- Detect human body keypoints from an image.
- Estimate body measurements using pose keypoints.
- Convert pixel measurements into approximate centimetre measurements.
- Predict an experimental clothing size.
- Provide the AI functionality through a Flask API for website and mobile application integration.

## Technologies Used

- Python
- Machine Learning
- Deep Learning
- Computer Vision
- YOLO Pose Estimation
- CNN
- Autoencoder
- LSTM
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- PyTorch
- Flask
- Flutter/Dart
- HTML, CSS and JavaScript
- SQLite

## AI Pipeline

User Image  
↓  
YOLO Pose Estimation  
↓  
Body Keypoint Detection  
↓  
Measurement Feature Extraction  
↓  
Pixel-to-Centimetre Calibration  
↓  
Body Measurements  
↓  
LSTM Size Prediction  
↓  
Clothing Size Recommendation

## Body Measurements

The current prototype estimates:

- Height
- Shoulder Width
- Hip Width
- Arm Length
- Leg Length

The measurements are approximate and can be affected by camera perspective, body pose, clothing, distance and image quality.

## Deep Learning Components

### YOLO Pose Estimation

YOLO Pose is used to detect the person and extract 17 human body keypoints.

### Autoencoder

A convolutional autoencoder was explored for image reconstruction and denoising. It is maintained as an experimental component and is not currently placed before YOLO because the reconstructed images reduced pose-detection reliability.

### LSTM

An LSTM-based prototype is used for experimental clothing-size prediction from body measurement features.

## Machine Learning

Machine learning models were also evaluated as baseline approaches for clothing-size prediction using anthropometric features.

## Flask API

The AI pipeline is exposed through a Flask REST API.

### Endpoint

POST /predict

The API accepts:

- User image
- User height in centimetres

and returns:

- Estimated body measurements
- Experimental clothing-size recommendation

## Project Structure

Smart-tailor-Ai/
│
├── ai-model/
│   ├── preprocessing/
│   ├── yolo-pose/
│   ├── measurement/
│   └── size-prediction/
│
├── dataset/
│   ├── raw/
│   ├── cleaned/
│   └── processed/
│
├── results/
├── backend/
├── mobile-app/
├── website/
│
├── .gitignore
└── README.md

## Current Status

The AI prototype currently supports:

1. Human pose detection using YOLO Pose.
2. Body keypoint extraction.
3. Approximate body measurement calculation.
4. Pixel-to-centimetre calibration.
5. Experimental clothing-size prediction using LSTM.
6. Flask API integration for connecting the AI system with the application.

## Limitations

The current system is a prototype. Measurements obtained from ordinary 2D images are approximate and can be affected by camera perspective, body pose, clothing, distance and image quality.

The current clothing-size prediction model is experimental and requires a larger and more representative real-world anthropometric dataset for reliable size recommendation.

## Future Scope

- Multi-view images of the same person.
- Improved camera calibration.
- Larger anthropometric datasets.
- Confidence estimation for measurements.
- Improved clothing-size prediction.
- Circumference measurement estimation.
- Mobile application integration.
- Tailor-side web dashboard.

## Team Project

Smart-Tailor-AI is being developed as a team project integrating AI/ML, backend services, website and mobile application components.
