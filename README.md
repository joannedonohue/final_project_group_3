# final_project_group_3

## Option 1: Credit Card Fraud Detection - focus on this one!
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Accompanying youtube:
https://www.youtube.com/watch?v=M_Cu7r9gik4

https://www.kaggle.com/datasets/kartik2112/fraud-detection

## Option 2: Also CC Fraud Detection - sounds like we won't use this one
https://www.kaggle.com/datasets/ealaxi/paysim1?resource=download

## Option 3: NLP w/ Amazon Reviews - Vivin to work on this 
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

