import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import time
import joblib
from pre_process import check_and_remove_nulls, preprocess_dataset
import os

def train_and_evaluate_models(input_path, processed_data_path, test_size=0.2, random_state=28, save_model=False):
    """
    Function to train and evaluate sentiment analysis model using Logistic Regression and save the model.
    
    Parameters:
    - input_path (str): Path to the raw dataset file.
    - processed_data_path (str): Path to the preprocessed data file.
    - test_size (float): Fraction of data to use for testing.
    - random_state (int): Random state for reproducibility.
    - save_model (bool): Whether to save the trained model and vectorizer to disk.
    
    Returns:
    - results (pd.DataFrame): DataFrame with training and testing accuracies for each model.
    """
    if os.path.exists(processed_data_path):
        print(f"Loading preprocessed data from {processed_data_path}...")
        df = pd.read_csv(processed_data_path, encoding='latin-1')
    else:
        print(f"Preprocessing raw data from {input_path}...")
        df = preprocess_dataset(input_path, save_path=processed_data_path)

    print(f"Data shape: {df.shape}")
    df = check_and_remove_nulls(df)
    print(f"Data shape after removing null values: {df.shape}")

    X = df['text']
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)

    print("Training Logistic Regression...")
    start_time = time.time()
    model.fit(X_train_vectorized, y_train)
    end_time = time.time()

    training_time = end_time - start_time

    y_train_pred = model.predict(X_train_vectorized)
    y_test_pred = model.predict(X_test_vectorized)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"Logistic Regression - Train Accuracy: {train_accuracy * 100:.2f}%, Test Accuracy: {test_accuracy * 100:.2f}%, Training Time: {training_time:.2f}s")

    if save_model:
        joblib.dump(model, 'model/logistic_regression_model.pkl')
        joblib.dump(vectorizer, 'model/tfidf_vectorizer.pkl')
        print("Model and vectorizer saved to disk.")

    return {
        'Model': 'Logistic Regression',
        'Train Accuracy': train_accuracy * 100,
        'Test Accuracy': test_accuracy * 100,
        'Training Time (seconds)': training_time,
    }

if __name__ == '__main__':
    input_path = 'sentiment140/training.1600000.processed.noemoticon.csv'  # Raw dataset path
    processed_data_path = 'sentiment140/processed_data.csv'  # Preprocessed dataset path
    results = train_and_evaluate_models(input_path, processed_data_path, save_model=True)
    print("\nModel Evaluation Results:")
    print(results)
