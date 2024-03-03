# Baitnet-url-detection

# BaitNet: Phishing Website Detection Using Deep Learning

BaitNet is a deep learning-based approach for detecting phishing websites, designed to provide robust protection against online threats. This repository contains the code implementation of the BaitNet model, along with instructions for usage and additional resources.

## Overview

Phishing attacks pose a significant threat to users' online security by attempting to deceive individuals into providing sensitive information such as login credentials, financial details, and personal data. BaitNet offers a solution by leveraging deep learning techniques to accurately identify phishing websites, thereby enhancing cybersecurity measures for users.

## Features

- Utilizes Convolutional Neural Networks (CNNs) for effective feature extraction from URL sequences
- Implements character-level tokenization for capturing intricate patterns in URLs
- Applies oversampling techniques to address class imbalance and improve model performance
- Incorporates Leaky ReLU activation and Dropout layers to enhance model robustness and prevent overfitting
- Achieves high accuracy and reliability in detecting malicious websites, as demonstrated in the research paper

## Usage

1. Clone the repository to your local machine:

```bash
git clone https://github.com/IPHive-24/Baitnet-url-detection.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Train the BaitNet model on your dataset or use the pre-trained model for inference.

## Dataset

The BaitNet model was trained on a dataset comprising X phishing URLs and Y legitimate URLs. Due to data privacy and licensing restrictions, the dataset used for training is not provided in this repository. However, users can train the model on their own datasets or utilize publicly available datasets for experimentation.

## Results

The effectiveness of the BaitNet model is demonstrated through comprehensive evaluation metrics, including accuracy, precision, recall, F1-score, sensitivity, specificity. Refer to the research paper for detailed results and comparative analysis.

## Contributions

Contributions to the BaitNet project are welcome! If you have any ideas for improvements, bug fixes, or new features, feel free to submit a pull request or open an issue.

## Acknowledgments

- This work is based on the research paper titled [BaitNet: A Deep Learning Approach for Phishing Detection](https://ieeexplore.ieee.org/document/10436016)
- Special thanks to our Mentor Dr.N.Nanthini for guidance and support throughout the project.
