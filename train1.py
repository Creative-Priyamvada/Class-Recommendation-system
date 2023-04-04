import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib

# Load the dataset
data = pd.read_json('idmanual.json')

# Preprocess the data
data['description'] = data['description'].str.lower().str.replace('[^\w\s]','')

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Extract features using TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
train_tfidf_matrix = tfidf_vectorizer.fit_transform(train_data['description'])

# Compute cosine similarities between items
cosine_similarities = cosine_similarity(train_tfidf_matrix, train_tfidf_matrix)

'''# Define a function to recommend classes for a given description
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
'''


# Save the trained model
try:
    print(' -- try --')
    joblib.dump(tfidf_vectorizer, 'data/tfidf_vectorizer.joblib')
    joblib.dump(train_tfidf_matrix, 'data/train_tfidf_matrix.joblib')
    joblib.dump(cosine_similarities, 'data/cosine_similarities.joblib')
    joblib.dump(train_data, 'data/train_data.joblib')
    print(' --- fin ---')
except Exception as e:
    print('-- e --')
    print(e)