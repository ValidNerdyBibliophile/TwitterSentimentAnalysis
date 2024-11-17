# Twitter Sentiment Analysis

This repository contains a complete pipeline for sentiment analysis of tweets, using the **Sentiment140 dataset**. The project leverages natural language processing (NLP) techniques and machine learning to predict whether a tweet has **positive** or **negative sentiment**, achieving an accuracy of **77%** using Logistic Regression.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Setup and Installation](#setup-and-installation)
4. [Pipeline Details](#pipeline-details)
    - [Dataset](#dataset)
    - [Preprocessing](#preprocessing)
    - [Model Training](#model-training)
    - [Prediction API](#prediction-api)
5. [Usage Instructions](#usage-instructions)
6. [Results](#results)
7. [Future Improvements](#future-improvements)
8. [Acknowledgments](#acknowledgments)

---

## Project Overview

This project implements a complete **end-to-end solution** for sentiment analysis of tweets. The key features include:
- Preprocessing and cleaning of raw data from Sentiment140.
- Training and evaluation of a Logistic Regression model with **TF-IDF vectorization**.
- A FastAPI-powered web API for real-time sentiment prediction.
- Visualization of the target column distribution.

---

## Directory Structure

The project directory is organized as follows:

```
.
├── core
│   ├── pre_process.py          # Preprocessing functions
│   ├── train_model.py          # Model training and evaluation logic
├── images
│   └── target_colum_distribution.png # Visualization of target column distribution
├── main.py                     # FastAPI implementation for prediction API
├── model
│   ├── logistic_regression_model.pkl # Trained Logistic Regression model
│   └── tfidf_vectorizer.pkl          # Trained TF-IDF vectorizer
├── requirements.txt            # Python dependencies
├── sentiment140                # Dataset
│   ├── processed_data.csv      # Preprocessed dataset
│   └── training.1600000.processed.noemoticon.csv # Raw dataset from Sentiment140
├── static
│   └── index.html              # HTML UI for the API
└── utils
    └── get_dataset_from_kaggle.py  # Script to download Sentiment140 dataset
```

---

## Setup and Installation

### Prerequisites

- Python 3.12 or higher
- A virtual environment (recommended)
- Kaggle API credentials

### Steps

1. Clone this repository:
   ```bash
   git clone https://github.com/username/TwitterSentimentAnalysis.git
   cd TwitterSentimentAnalysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the Sentiment140 dataset:
   - Place your `kaggle.json` file in the `utils/.kaggle` directory.
   - Run the dataset download script:
     ```bash
     python utils/get_dataset_from_kaggle.py
     ```

4. Preprocess the dataset:
   ```bash
   python core/pre_process.py
   ```

5. Train the model:
   ```bash
   python core/train_model.py
   ```

6. Start the API server:
   ```bash
   python main.py
   ```

7. Access the web interface:
   - Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## Pipeline Details

### Dataset

The project uses the **Sentiment140 dataset**, which contains **1.6 million tweets** labeled with:
- `0`: Negative sentiment
- `4`: Positive sentiment

### Preprocessing

The preprocessing script (`pre_process.py`) performs the following steps:
1. Removes URLs, mentions, hashtags, and non-alphabetical characters.
2. Converts text to lowercase.
3. Removes stopwords using NLTK's stopword list.
4. Applies stemming using the Porter Stemmer.

### Model Training

- **TF-IDF Vectorizer** is used to convert text data into numerical features.
- **Logistic Regression** is employed for classification.
- The training and testing split ratio is **80:20**.
- Results:
  - **Training Accuracy:** 79%
  - **Testing Accuracy:** 77%
- The trained model and vectorizer are saved in the `model/` directory.

### Prediction API

The `main.py` script implements a **FastAPI** server with:
1. **GET `/`**: Serves the HTML UI for user input.
2. **POST `/predict`**: Accepts text input and returns sentiment prediction as JSON.

---

## Usage Instructions

### Predict Sentiment Using API

1. Start the server:
   ```bash
   python main.py
   ```

2. Open the web interface or send a POST request to `/predict` with the following format:
   ```json
   {
     "text": "I love this movie!"
   }
   ```

3. Example Response:
   ```json
   {
     "sentiment": "Positive"
   }
   ```

---

## Results

- **Target Column Distribution:**
  ![Distribution of Target Column](images/target_colum_distribution.png)

- **Model Performance:**
  - Train Accuracy: 79%
  - Test Accuracy: 77%
  - Training Time: ~30 seconds (depends on hardware).

---

## Future Improvements

Here are some potential enhancements that can be made to the project:

### 1. Improved Model Accuracy
- **Advanced Models:** Implement deep learning models like LSTMs, GRUs, or BERT for better contextual understanding of the text.
- **Hyperparameter Tuning:** Perform grid search or random search to optimize the Logistic Regression hyperparameters further.
- **Ensemble Models:** Use ensemble methods like Random Forests or XGBoost to combine multiple models for more robust predictions.

### 2. Enhanced Data Preprocessing
- **Advanced Text Cleaning:** Add lemmatization along with stemming for better generalization of word forms.
- **Spell Correction:** Include a spell checker to correct typos and improve the quality of input data.
- **Handle Emojis:** Map emojis to sentiments using pre-defined dictionaries for additional sentiment cues.
- **Handling Slang and Abbreviations:** Use a library or custom dictionary to translate common slang and abbreviations into formal text.

### 3. Expanded Dataset
- **Multilingual Support:** Extend the model to analyze tweets in multiple languages by translating them to English or training on multilingual datasets.
- **Augmented Dataset:** Use data augmentation techniques like synonym replacement or paraphrasing to expand the dataset.
- **Dynamic Dataset:** Continuously update the dataset with new tweets to keep the model up to date with evolving language trends.

### 4. Model Interpretability
- **Explainable AI:** Integrate tools like SHAP or LIME to explain why the model predicts a specific sentiment.
- **Keyword Highlights:** Highlight the words in the input text that contributed most to the sentiment score.

### 5. User Interface Improvements
- **Mobile Compatibility:** Make the web app fully responsive and compatible with mobile devices.
- **Real-Time Analysis:** Fetch live tweets using the Twitter API and display sentiment analysis results dynamically.
- **Visualization:** Provide sentiment distribution charts, word clouds, or timelines of sentiments for better insights.

### 6. Scalability and Deployment
- **Cloud Deployment:** Deploy the application to cloud platforms like AWS, GCP, or Azure for better scalability.
- **Containerization:** Use Docker to containerize the application for easier deployment and portability.
- **Load Balancing:** Add load balancers to handle high traffic efficiently.

### 7. Fine-Grained Sentiment Analysis
- **Multi-Class Sentiments:** Extend the analysis to classify tweets into more granular sentiment categories (e.g., very positive, somewhat positive, neutral, etc.).
- **Aspect-Based Sentiment Analysis:** Analyze sentiments for specific topics or aspects within tweets (e.g., "battery life" in a product review).

### 8. Feedback Loop
- **User Feedback:** Allow users to provide feedback on incorrect predictions and use this feedback to retrain and improve the model.
- **Active Learning:** Implement active learning to retrain the model incrementally using new user-provided data.

### 9. Ethical Considerations
- **Bias Mitigation:** Analyze the model for any biases and retrain it to avoid unfair predictions.
- **Data Privacy:** Ensure compliance with data privacy laws like GDPR when collecting and analyzing tweets.

### 10. Additional Features
- **Sentiment Over Time:** Analyze sentiment trends over a period to identify patterns.
- **Topic Modeling:** Use NLP techniques like Latent Dirichlet Allocation (LDA) to identify common topics in tweets.
- **Sarcasm Detection:** Include a module to identify sarcastic tweets, which are often challenging for sentiment analysis models.

These improvements can make your project more robust, scalable, and feature-rich, addressing both user experience and analytical depth.

---

## Acknowledgments

- The **Sentiment140 dataset** was created by Alec Go, Richa Bhayani, and Lei Huang.
- Thanks to **Kaggle** for providing easy access to the dataset.
