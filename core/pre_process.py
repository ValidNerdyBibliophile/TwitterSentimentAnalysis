import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import os

# Initialize stemmer and stopwords globally
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Function to preprocess dataset
def preprocess_dataset(input_path, save_path=None):
    try:
        # Read the raw data
        columns = ['target', 'id', 'date', 'flag', 'user', 'text']
        df = pd.read_csv(input_path, encoding='latin-1', names=columns)  # Read raw dataset
        df = check_and_remove_nulls(df)  # Check and remove null values
        print("Cleaning the text.")
        df['text'] = df['text'].apply(lambda x: stemm_text(x))  # Apply stemming to text column
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Processed data saved to {save_path}.")
        return df
    except Exception as e:
        print(f"Error in preprocessing dataset: {e}")
        return None

# Function to check for and remove null values
def check_and_remove_nulls(df):
    print("Checking for null values...")
    null_count = df.isnull().sum()
    # print(null_count)
    if null_count.any():
        print("Null values found. Removing rows with null values...")
        df = df.dropna()
    else:
        print("No null values found.")
    return df

# Function to check the target column distribution (optional)
def check_target_distribution(df):
    print("Checking distribution of target column...")
    distribution = df['target'].value_counts()
    print(distribution)
    
    plt.figure(figsize=(6, 4))
    distribution.plot(kind='bar', color=['blue', 'orange'])
    plt.title('Distribution of Target Column')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks([0, 1], ['Negative (0)', 'Positive (4)'], rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    return distribution

# Function to preprocess and stem text
def stemm_text(text):
    # Remove URLs, mentions, hashtags, and non-alphabetical characters
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove non-alphabetical characters
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)


if __name__ == '__main__':
    input_path = 'sentiment140/training.1600000.processed.noemoticon.csv'
    save_path = 'sentiment140/processed_data.csv'

    if not os.path.exists(save_path):
        df = preprocess_dataset(input_path, save_path)
    else:
        print(f"Loading preprocessed data from {save_path}...")
        df = pd.read_csv(save_path)
