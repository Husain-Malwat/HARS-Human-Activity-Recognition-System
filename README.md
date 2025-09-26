# Human Activity Recognition (HAR) Mini Project

## Overview

Implementation of a Human Activity Recognition system using accelerometer and gyroscope data. The dataset used is the UCI-HAR dataset, where participants performed six activities. The goal is to analyze and classify activities based on sensor data.

## Dataset

- [UCI-HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- Raw accelerometer data from the inertial_signals folder used.
- Data organized and sorted using CombineScript.py and MakeDataset.py scripts.

## Preprocessing

1. Place CombineScript.py and MakeDataset.py in the UCI dataset folder.
2. Run CombineScript.py to organize data into the Combined folder.
3. Run MakeDataset.py to create a dataset with train, test, and validation sets.
4. Focus on the initial 10 seconds of activity (500 data samples at 50Hz).

## Tasks

### 1. Waveform Analysis

- Plot waveform for each activity class.
- Observe differences/similarities in a subplot with 6 columns.

### 2. Static vs. Dynamic Activities

- Analyze linear acceleration for each activity.
- Justify the need for a machine learning model to differentiate static and dynamic activities.

### 3. Decision Tree Training and Evaluation

- Train Decision Tree using the train set.
- Report accuracy and confusion matrix using the test set.
- Train Decision Tree with varying depths (2-8) and analyze accuracy changes.

### 4. Feature Engineering and Visualization

- Use PCA on Total Acceleration for dimensionality reduction.
- Apply TSFEL for feature extraction.
- Visualize different activity classes using scatter plots.

### 5. Decision Tree with Engineered Features

- Train Decision Tree using features from TSFEL.
- Report accuracy and confusion matrix using the test set.
- Compare Decision Tree accuracies with varying depths using raw data and engineered features.

### 6. Model Performance Analysis

- Identify participants/activities with poor model performance.
- Analyze reasons for performance issues.

### 7. Deployment

- Utilize Physics Toolbox Suite to collect smartphone sensor data.
- Trim data to 10 seconds, ensuring consistent phone position and alignment.
- Train on UCI dataset and test on collected data.
- Report accuracy and confusion matrix.

## Results

### Confusion Matrix

### Scatter Plots

- PCA Scatter Plot: Limited separability.
- TSFEL + PCA Scatter Plot: Improved class separability.

### Decision Tree Depth Impact

- Optimal depth crucial for balancing bias-variance tradeoff.

### Model Deployment

- Successful classification in real-world scenarios.
- Identified special cases impacting model performance.

## Future Improvements

- Explore ensemble methods for dynamic activities.
- Investigate advanced feature engineering techniques.
- Implement real-time monitoring for dynamic environments.
