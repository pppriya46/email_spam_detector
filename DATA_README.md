# Dataset Information

## Overview
Due to file size limitations on GitHub, the actual dataset CSV files are not included in this repository.

## Required Datasets

### 1. Spam_Email_raw_text_for_NLP.csv
- **Source:** Kaggle
- **Size:** ~16 MB
- **Samples:** ~5,000+
- **Format:** Raw email text with labels

### 2. mail.csv
- **Source:** Kaggle
- **Size:** ~15 MB
- **Samples:** ~5,700+
- **Format:** Preprocessed email messages

### 3. SpamAssassin.csv
- **Source:** Apache SpamAssassin Public Corpus
- **Size:** ~4 MB
- **Samples:** ~3,000
- **Format:** Clean labeled emails

## Final Dataset
After preprocessing and balancing:
- **Total Samples:** 8,284
- **Spam:** 4,142 (50%)
- **Ham:** 4,142 (50%)
- **Features:** Preprocessed text messages

## How to Get Datasets
1. Visit Kaggle and search for email spam datasets
2. Download the three datasets mentioned above
3. Place them in a `data/` folder in your local clone
4. Run `dataset_spam_mail_classification.ipynb` to preprocess

## Note
The notebooks are configured to load data from Google Drive when running in Colab.
Update file paths accordingly for local execution.
