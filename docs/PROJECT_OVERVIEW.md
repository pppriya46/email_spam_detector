# Project Overview: Spam Mail Classification

## Problem Statement
Build and optimize a hybrid deep learning model for binary classification of emails as spam or not spam (ham).

## Methodology

### 1. Data Collection
- Collected 3 datasets from Kaggle and public sources
- Total raw samples: ~10,000+ emails

### 2. Data Preprocessing
- Removed HTML tags using BeautifulSoup
- Removed URLs and links (http, www)
- Removed punctuation and special characters
- Converted text to lowercase
- Removed stopwords
- Balanced classes to 50-50 distribution

### 3. Data Encoding
- Tokenization with 20,000 word vocabulary
- Sequence padding/truncating to maxlen=150
- Train-test split: 80-20

### 4. Model Architecture
- Hybrid RNN with cuDNNLSTM and cuDNNGRU layers
- GPU-accelerated using NVIDIA cuDNN
- Dropout regularization
- Binary classification output

### 5. Training & Optimization
- Adam optimizer
- Binary crossentropy loss
- Early stopping callback
- Model checkpointing

### 6. Evaluation
- Confusion matrix analysis
- Precision, Recall, F1-score
- Benchmark on separate test data

## Key Features
- GPU acceleration for faster training
- Hybrid architecture combining LSTM and GRU
- Balanced dataset for unbiased learning
- High accuracy (>99%) on test data
