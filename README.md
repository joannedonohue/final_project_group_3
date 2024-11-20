# final_project_group_3

# Dataset 1: Credit Card Fraud Detection
https://www.kaggle.com/datasets/kartik2112/fraud-detection

## Project Overview
# Advanced Machine Learning for Credit Card Fraud Detection

This repository contains a comprehensive analysis and implementation of advanced machine learning models, including transformers, TabNet, and neural networks, to address the critical challenge of detecting fraudulent credit card transactions.

## Table of Contents
- [Problem Definition](#problem-definition)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Development](#model-development)
  - [Random Forest](#random-forest)
  - [TabNet](#tabnet)
  - [Neural Networks](#neural-networks)
- [Model Evaluation](#model-evaluation)
- [Confusion Matrix Analysis](#confusion-matrix-analysis)
- [Conclusion](#conclusion)

---

## Problem Definition
Detecting credit card fraud is a persistent challenge in the finance sector. This project aims to build a scalable and reliable fraud detection system capable of identifying fraudulent transactions with minimal disruption to legitimate activities. The project addresses key challenges such as:
- **Class imbalance** in datasets.
- **Dynamic fraudulent patterns** in data.
- **Real-time detection** requirements.

We leverage advanced machine learning techniques, including:
- TabNet for tabular data.
- Transformers for contextual embeddings.
- Neural networks for deeper pattern recognition.

---

## Dataset
- **Source:** [Simulated Credit Card Fraud Dataset](#)
- **Training Data:** `fraudTrain.csv`
- **Test Data:** `fraudTest.csv`
- **Size:**
  - Training Data: 1,296,675 rows × 17 columns
  - Test Data: 555,719 rows × 17 columns
- **Target Variable:** `is_fraud` (0: Non-Fraud, 1: Fraud)

---

## Data Preprocessing
### Steps:
1. **Dropped irrelevant columns** like personal identifiers.
2. **Engineered features** such as transaction time, day, month, and customer age.
3. **Encoded categorical variables** using `LabelEncoder`.
4. **Standardized numerical features** using `StandardScaler`.
5. **Balanced the dataset** using SMOTE to address class imbalance.

### Results:
- Complete datasets with no missing values.
- Preprocessed features and labels ready for model training and evaluation.

---

## Exploratory Data Analysis (EDA)
### Key Insights:
1. **Class Imbalance:** Fraudulent transactions constitute a small fraction of the dataset.
2. **Transaction Amount Distribution:** Fraudulent transactions often involve higher amounts.
3. **Category Analysis:** Certain merchant categories have higher fraud rates.
4. **Correlation Analysis:** Weak correlations among numerical features suggest the need for feature engineering.

### Visualizations:
- Fraud vs. Non-Fraud distribution
- Fraud rates by category
- Transaction amount distributions
- Correlation heatmaps

---

## Model Development
### 1. Random Forest
- **Baseline Model:** Trained on a balanced dataset using SMOTE.
- **Performance Metrics:**
  - Accuracy: `0.9973`
  - Precision: `0.6567`
  - Recall: `0.6448`
  - F1-Score: `0.6507`
  - AUC-ROC: `0.8217`

### 2. TabNet
- **Advanced Model:** TabNet architecture specialized for tabular data.
- **Performance Metrics:**
  - Accuracy: `0.9512`
  - Precision: `0.0667`
  - Recall: `0.8956`
  - F1-Score: `0.1241`
  - AUC-ROC: `0.9235`

### 3. Neural Networks
- **Architecture:**
  - Input Layer
  - Two Hidden Layers with ReLU activation
  - Output Layer with Sigmoid activation
- **Performance Metrics:**
  - Accuracy: `0.9703`
  - Precision: `0.0645`
  - Recall: `0.4956`
  - F1-Score: `0.1142`
  - AUC-ROC: `0.7339`

---

## Model Evaluation
All models were evaluated on the test dataset using the following metrics:
1. **Accuracy:** Proportion of correctly classified instances.
2. **Precision:** Fraction of correctly identified fraud cases out of all predicted fraud cases.
3. **Recall:** Fraction of actual fraud cases correctly identified.
4. **F1-Score:** Harmonic mean of precision and recall.
5. **AUC-ROC:** Area under the receiver operating characteristic curve.

---

## Confusion Matrix Analysis
- Visualized confusion matrices for:
  - **Random Forest**
  - **TabNet**
  - **Neural Networks**

Key insights:
- Random Forest balanced false positives and false negatives well.
- TabNet demonstrated high recall but struggled with precision, leading to many false positives.
- Neural Networks performed moderately but require further optimization to improve precision and recall.

---

## Conclusion
### Summary of Results:
- **Random Forest** was the most balanced model, excelling in accuracy and F1-score.
- **TabNet** showed promise with a high AUC-ROC but suffered from low precision.
- **Neural Networks** require further fine-tuning to achieve better performance.

### Future Work:
1. Improve **precision** for TabNet and Neural Networks to reduce false positives.
2. Explore advanced transformer-based architectures for tabular data.
3. Refine feature engineering techniques to enhance model performance.
4. Experiment with real-time fraud detection techniques.

---

# Dataset 2: Credit Card Fraud Detection
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Project Overview
# Credit Card Fraud Detection

This repository contains a Python project to analyze and build machine learning models for detecting fraudulent credit card transactions. The dataset used is highly imbalanced and consists of numerical features transformed via PCA for confidentiality.

## Dataset Overview

- **Source**: [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: 284,807 rows × 31 columns
- **Target**: `Class` (0: Non-Fraud, 1: Fraud)
- **Features**: 28 PCA-transformed features, `Time`, and `Amount`.

## Project Workflow

### 1. Data Importation and Preparation
- Dataset loaded using Kaggle API and preprocessed in Google Colab.
- Features like `Amount` and `Time` were scaled/normalized using `RobustScaler`.
- Dataset split into training, validation, and testing sets.
- Due to severe class imbalance, models were trained on both original and balanced datasets.

### 2. Exploratory Data Analysis
- Visualized feature distributions using histograms.
- Observed imbalanced class distribution.
- Extracted summary statistics for key insights into feature ranges and variances.

### 3. Models Implemented
#### Unbalanced Dataset
1. **Logistic Regression**
2. **Shallow Neural Network**
3. **Random Forest**
4. **Gradient Boost**
5. **Linear SVC**

#### Balanced Dataset
1. **Logistic Regression**
2. **Shallow Neural Network (Single/Double Hidden Layers)**
3. **Random Forest**
4. **Gradient Boost**
5. **Linear SVC**

### 4. Results and Observations
- Models performed better on the balanced dataset, improving fraud detection.
- Performance metrics (accuracy, F1-score, precision, recall) calculated for all models.
- Shallow Neural Network and Gradient Boost performed well but required tuning to mitigate overfitting.

### 5. Key Insights
- Class imbalance significantly impacts model performance.
- Balancing the dataset via undersampling improved fraud detection.
- Model complexity and dataset size need careful balancing to avoid overfitting.


# Dataset 3: NLP w/ Amazon Reviews
# Amazon Reviews Sentiment and Aspect-Based Analysis

## Project Overview
This project analyzes Amazon reviews to:
- Perform **sentiment analysis** using VADER.
- Extract **aspect-based sentiments** (e.g., design, price).
- Implement machine learning models (Logistic Regression) for sentiment classification.
- Apply **topic modeling** with LDA to uncover hidden themes in reviews.
- Visualize findings using word clouds, bar charts, and interactive tools like `pyLDAvis`.

---

## Key Features
1. **Sentiment Analysis**:
   - Calculates sentiment scores (positive/negative) for reviews using the VADER sentiment analyzer.
   - Handles multilingual reviews and missing values.

2. **Aspect-Based Sentiment Analysis**:
   - Identifies sentiment related to specific product aspects like "design," "durability," "price," and "brand."
   - Uses `TextBlob` to extract polarity (positive/negative).

3. **Machine Learning**:
   - Converts review text into numerical representations using TF-IDF vectorization.
   - Trains a Logistic Regression model with hyperparameter tuning using `GridSearchCV`.
   - Evaluates model performance with metrics such as Accuracy, Precision, Recall, and F1-score.

4. **Topic Modeling**:
   - Implements Latent Dirichlet Allocation (LDA) to discover hidden topics in reviews.
   - Visualizes topic distributions and relevance using `pyLDAvis`.

5. **Data Visualizations**:
   - **Sentiment distribution**: Bar chart of positive and negative review counts.
   - **Word cloud**: Highlights frequently mentioned terms in negative reviews.
   - **Aspect sentiment distribution**: Bar chart comparing sentiment counts for key product aspects.

---

## Technologies Used
- **Python Libraries**:
  - `pandas`, `numpy`: Data handling and processing.
  - `VADER` from Nltp
  - `nltk`, `TextBlob`: Natural Language Processing for sentiment and polarity analysis.
  - `scikit-learn`: Machine learning, TF-IDF, and evaluation metrics.
  - `matplotlib`, `seaborn`: Data visualizations.
  - `wordcloud`: Generate word clouds.
  - `pyLDAvis`: Interactive topic modeling visualization.
- **Google Colab**:
  - Access to Google Drive for data storage and processing.

---

## Project Workflow
1. **Data Preprocessing**:
   - Load Amazon reviews dataset (`.tsv` format) from Google Drive.
   - Clean and preprocess data by handling missing values and irrelevant content.

2. **Sentiment Analysis**:
   - Compute sentiment scores for each review using VADER.
   - Categorize reviews as positive or negative based on compound sentiment scores.

3. **Aspect-Based Analysis**:
   - Extract and classify sentiments for specific product aspects.

4. **Machine Learning**:
   - Train-test split on the dataset.
   - Use TF-IDF for feature extraction.
   - Train a Logistic Regression model with optimized parameters.
   - Evaluate the model using Accuracy, Precision, Recall, and F1-score.

5. **Topic Modeling**:
   - Run LDA to extract hidden topics from reviews.
   - Visualize topics and associated terms using `pyLDAvis`.

6. **Visualizations**:
   - Display results with bar charts and word clouds for interpretability.

---

## Setup Instructions

### Prerequisites
- Python 3.x
- Libraries: Install the required dependencies using:
  ```bash
  pip install pandas numpy nltk scikit-learn matplotlib seaborn wordcloud pyLDAvis tqdm

