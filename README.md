# Image Classification with TensorFlow - MNIST Handwritten Digits - 

This project trains and evaluates a neural network to recognize handwritten digits (0–9) from grayscale images using the classic **MNIST** dataset.

The goal is to build a complete, reproducible pipeline:

- Load and explore image data
- Preprocess and normalize pixel values
- One-hot encode labels for multi-class classification
- Build and train a feed-forward neural network in TensorFlow/Keras
- Monitor training with validation curves and callbacks
- Evaluate performance on a held-out test set
- Analyze errors with a confusion matrix and classification report
- Visualize example predictions

Although MNIST is a standard dataset, this project demonstrates a full deep-learning workflow for **image classification**, which is directly relevant to roles involving image-based search, user experience metrics, and model-driven product decisions.

## 1. Project Overview

**Problem**  
Given a 28×28 grayscale image of a handwritten digit, predict which digit (0–9) it represents.

- **Input:** 28×28 pixel image (grayscale)
- **Output:** One of 10 classes (digits 0–9)
- **Task type:** Supervised multi-class classification

The model learns from 60,000 labeled training images and is evaluated on 10,000 test images.

## 2. Tech Stack

- **Language:** Python 3.x  
- **Core Libraries:**
  - [`tensorflow` / `keras`](https://www.tensorflow.org/) – model definition, training, evaluation
  - `numpy` – numerical operations
  - `matplotlib` – visualization (images, loss/accuracy curves)
  - `seaborn` – confusion matrix heatmap
  - `scikit-learn` – confusion matrix & classification report

## 3. Dataset (MNIST)

The project uses the MNIST dataset via:

```python
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

Training set: 60,000 images (28×28)
Test set: 10,000 images (28×28)
Pixel values: 0–255 (grayscale)
Labels: Integers 0–9

## 4. Preprocessing

• Normalize Images: Each 28×28 image is reshaped into a 784-dimensional vector and scaled to [0, 1]. This converts each image to a feature vector suitable for a dense neural network. Normalizes pixel intensities to stabilize and speed up training.

• One-Hot Encode Labels: Convert integer labels into one-hot vectors for use with categorical_crossentropy.



This project demonstrates practical experience with image data, neural networks, and experimentation, which are foundational skills for roles focused on image-based search and user experience analytics.
