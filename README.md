# MFG598 Final Project

## Vegetation Detection Using KNN ML Algorithm

### Project Overview

This project focuses on the detection of vegetation areas using the K-Nearest Neighbors (KNN) machine learning algorithm. It employs Python and a set of powerful libraries to process image data and identify vegetative regions.

### Python Packages

The following Python packages are instrumental in this project:

- `Scikit-Learn`: Implements KNN and provides metrics for model evaluation.
- `OpenCV`: Facilitates image processing tasks.
- `Matplotlib`: Used for visualizing results through plots and graphs.
- `NumPy`: Handles numerical operations on array data structures.

### Scripts and Functionality

The repository is structured with two primary scripts:

- `train_data.py`: Collects and processes training data from two images, `train_1` and `train_2`, extracting essential color features for the KNN model.

- `train_validation.py`: Analyzes a validation image to determine vegetation coverage, trains the KNN model with the training data, and outputs the model's accuracy, confusion matrix, and the vegetation percentage of the validation image.

### Outputs

Execution of the scripts will yield:

- Model Accuracy: Quantitative measure of the KNN model performance.
- Confusion Matrix: Visual representation of the model's predictive accuracy.
- Vegetation Percentage: The determined proportion of vegetation in the validation image.

