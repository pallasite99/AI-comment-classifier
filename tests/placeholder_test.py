import pytest
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Mock data for testing
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'text': [
            "This is a test sentence.",
            "Another example text from an AI model.",
            "Humans write text like this.",
            "AI can generate sentences too.",
            "What makes AI different is its patterns."
        ],
        'generated': [0, 1, 0, 1, 1]  # 0 = Human, 1 = AI
    })


# Test for data preprocessing
def test_preprocessing(sample_data):
    # Example preprocessing function
    def clean_text(text):
        return text.lower().replace(".", "").replace(",", "")
    
    sample_data['cleaned_text'] = sample_data['text'].apply(clean_text)
    
    assert 'cleaned_text' in sample_data.columns
    assert sample_data['cleaned_text'][0] == "this is a test sentence"


# Test for TF-IDF vectorization
def test_tfidf_vectorization(sample_data):
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(sample_data['text'])
    
    assert X.shape[0] == len(sample_data)  # Check rows
    assert X.shape[1] <= 500  # Check feature limit


# Test for Logistic Regression training
def test_logistic_regression_training(sample_data):
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(sample_data['text'])
    y = sample_data['generated']
    
    # Train model
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X, y)
    
    # Ensure the model has been trained
    assert len(model.coef_[0]) == X.shape[1]


# Test for accuracy calculation
def test_model_accuracy(sample_data):
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(sample_data['text'])
    y = sample_data['generated']
    
    # Train model
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X, y)
    
    # Predict on the same data
    y_pred = model.predict(X)
    
    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)
    assert accuracy > 0.5  # Ensure accuracy is reasonable

