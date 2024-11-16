# Time Series Classification with Self-Supervised Learning

This project aims to improve time series classification accuracy by utilizing self-supervised learning (SSL) techniques. Specifically, it focuses on gesture recognition in the UWaveGestureLibrary dataset, leveraging feature representations learned from pretraining on a separate dataset (HAR). The project involves dataset preparation, SSL implementation, classification model training, and performance evaluation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Implementation Details](#implementation-details)
  - [Dataset Preparation](#dataset-preparation)
  - [Self-Supervised Learning](#self-supervised-learning)
  - [Classification Model](#classification-model)
  - [Performance Verification](#performance-verification)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)

---

## Project Overview

This project demonstrates how self-supervised learning can enhance time series classification accuracy. Self-supervised learning leverages unlabeled data to extract meaningful features, which are later fine-tuned on labeled data. The UWaveGestureLibrary dataset is used for gesture classification, with feature pretraining on the HAR dataset.

---

## Dataset

1. **UWaveGestureLibrary Dataset**: Used for gesture classification.
   - URL: [UWaveGestureLibrary Dataset](https://www.timeseriesclassification.com/description.php?Dataset=UWaveGestureLibrary)
   - Contains time series data of various gestures.

2. **HAR Dataset**: Used for self-supervised pretraining.
   - Contains sensor data for human activity recognition.

**File Structure**:
├── Gesture/ │ ├── train.pt │ ├── test.pt │ ├── val.pt ├── HAR/ │ ├── train.pt │ ├── test.pt │ ├── val.pt

yaml
Copy code

---

## Project Structure

. ├── main.py # Entry point of the project ├── data_utils.py # Handles dataset loading and preprocessing ├── ssl_model.py # Implementation of self-supervised learning ├── classifier_model.py # Implementation of the classification model ├── README.md # Project documentation (this file) └── requirements.txt # Python dependencies

yaml
Copy code

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository-url.git
   cd time-series-ssl
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Ensure you have the required datasets:

Place the Gesture and HAR directories in the project root.
Usage
Preprocess the datasets: Run the preprocessing script to clean and normalize the datasets.

bash
Copy code
python main.py
Train the SSL model: Pretrain the self-supervised learning model on the HAR dataset.

bash
Copy code
python main.py --task pretrain
Train the classifier: Fine-tune the classifier on the UWaveGestureLibrary dataset.

bash
Copy code
python main.py --task train_classifier
Evaluate the model: Evaluate the performance of the trained classifier.

bash
Copy code
python main.py --task evaluate
Implementation Details
Dataset Preparation
Preprocessed datasets by normalizing and handling missing values.
Data scaling with StandardScaler.
Self-Supervised Learning
Pretraining on the HAR dataset to learn feature representations.
Applied an SSL loss function (e.g., contrastive or triplet loss) to align similar time series samples.
Classification Model
Fine-tuned a neural network model (e.g., LSTM or CNN) using the UWaveGestureLibrary dataset.
Utilized the features extracted during the SSL stage to enhance performance.
Performance Verification
Evaluated classification accuracy with and without SSL.
Reported metrics:
Accuracy
Precision
Recall
F1-Score
Results
Baseline Accuracy: X% (without SSL)
Improved Accuracy: Y% (with SSL)
Performance Gain: Z%
The results indicate that pretraining with SSL significantly enhances classification accuracy for the UWaveGestureLibrary dataset.

Future Work
Experiment with advanced SSL techniques such as SimCLR or BYOL.
Test on additional time series datasets for generalization.
Optimize model architecture and hyperparameters for further improvement.
References
Self-Supervised Learning Papers
UWaveGestureLibrary Dataset: Link
HAR Dataset: Link
License
This project is licensed under the MIT License. See the LICENSE file for details.
