import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib


# Load the saved model
tfidf_vectorizer = joblib.load('data/tfidf_vectorizer.joblib')
train_tfidf_matrix = joblib.load('data/train_tfidf_matrix.joblib')
cosine_similarities = joblib.load('data/cosine_similarities.joblib')
train_data = joblib.load('data/train_data.joblib')

# Define a function to recommend classes using the saved model
def recommend_classes(description, top_n=5):
    # Preprocess the input
    description = description.lower().replace('[^\w\s]','')

    # Compute TF-IDF vectors for the input and training set
    input_tfidf = tfidf_vectorizer.transform([description])
    train_tfidf = train_tfidf_matrix

    # Compute cosine similarities between the input and training set
    cosine_similarities = cosine_similarity(input_tfidf, train_tfidf)[0]

    # Sort the similarities and retrieve the top N recommendations
    indices = np.argsort(cosine_similarities)[::-1][:top_n]
    recommendations = [(train_data.iloc[i]['class_id'], cosine_similarities[i]) for i in indices]

    return recommendations



# Test the recommendation function
description = 'Computer hardware, namely, electronic components'
recommendations = recommend_classes(description, top_n=5)
for recommendation in recommendations:
    print('Class ID: {}, Probability: {:.2%}'.format(recommendation[0], recommendation[1]))
