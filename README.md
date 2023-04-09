# Spam Mail Classification Using Deep Learning
## A Hybrid Multilayer RNN Approach

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📌 Project Overview
This project implements a deep learning model for binary classification of spam emails using a hybrid architecture combining different types of Recurrent Neural Networks (RNNs) including cuDNNLSTM and cuDNNGRU layers.

## 👥 Team Members
- **Shakti Ashutosh Panda** (1905804)
- **Shivaansh Sharma (Diba)** (1905808)

*Under the guidance of Dr. Debajyoty Banik*

## 🎯 Objectives
- Build a hybrid deep learning model for spam classification
- Preprocess and balance email datasets
- Achieve high accuracy using RNN architecture
- Optimize model using GPU acceleration (cuDNN)

## 📊 Dataset
- **Total Samples:** ~8,284 emails
- **Classes:** Balanced (50% spam, 50% ham)
- **Sources:** Kaggle datasets + SpamAssassin corpus

## 🛠️ Tech Stack
- Python 3.8+
- TensorFlow/Keras
- Google Colab (GPU runtime)
- Libraries: pandas, numpy, BeautifulSoup, re

## 📁 Project Structure
```
spam-mail-classification/
│
├── dataset_spam_mail_classification.ipynb    # Data preprocessing
├── spam_mail_classification.ipynb            # Main model
├── benchmark_spam_mail_classification.ipynb  # Benchmarking
├── confusion_matrix_benchmark.ipynb          # Evaluation
├── data_collection_notes.md                  # Dataset info
└── README.md
```

## 🚀 Status
✅ Completed - April 2023

---

*KIIT Deemed to be University*  
*School of Computer Engineering*  
*Bhubaneswar, Odisha - 751024*

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Google Colab (for GPU support) or local GPU with CUDA support
- Required libraries (see requirements.txt)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/spam-mail-classification.git
cd spam-mail-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open notebooks in Google Colab:
   - Upload notebooks to Google Drive
   - Open with Google Colab
   - Change Runtime type to GPU

### Usage

1. **Data Preprocessing:**
   Run `dataset_spam_mail_classification.ipynb` to:
   - Load and merge datasets
   - Clean and preprocess text
   - Create balanced dataset

2. **Model Training:**
   Run `spam_mail_classification.ipynb` to:
   - Build the hybrid RNN model
   - Train on preprocessed data
   - Evaluate performance

3. **Benchmarking:**
   Run `benchmark_spam_mail_classification.ipynb` to:
   - Test model on new data
   - Generate predictions

4. **Evaluation:**
   Run `confusion_matrix_benchmark.ipynb` to:
   - Analyze confusion matrix
   - Calculate metrics

## 📈 Model Architecture

- **Embedding Layer:** 20,000 vocabulary, 128 dimensions
- **cuDNNLSTM Layers:** Multiple stacked layers
- **cuDNNGRU Layers:** Hybrid architecture
- **Dropout:** Regularization (0.2-0.5)
- **Dense Output:** Sigmoid activation for binary classification
- **Optimizer:** Adam
- **Loss:** Binary crossentropy

## 📊 Results

- **Accuracy:** ~99% on balanced dataset
- **Precision:** High spam detection rate
- **Recall:** Minimal false negatives
- **F1-Score:** Excellent overall performance


## 🤝 Contributing
This is an academic project. For major changes, please open an issue first to discuss proposed changes.

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍🎓 Academic Context
- **Institution:** KIIT Deemed to be University
- **Department:** School of Computer Engineering
- **Course:** Bachelor's Degree in Computer Science & Engineering
- **Semester:** 6th Semester (Spring 2023)
- **Supervisor:** Dr. Debajyoty Banik

## 📚 References
- Keras Documentation: https://keras.io/
- TensorFlow Guide: https://www.tensorflow.org/
- cuDNN Documentation: https://docs.nvidia.com/deeplearning/cudnn/

## 📧 Contact
- **Shakti Ashutosh Panda:** shakti.panda@example.com
- **Shivaansh Sharma:** shivaansh.sharma@example.com

---

**Note:** Dataset files are not included in the repository due to size constraints. They can be downloaded from Kaggle.

*Last Updated: April 2023*
