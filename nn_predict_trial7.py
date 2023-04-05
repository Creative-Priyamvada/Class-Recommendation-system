import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# Load the saved model
model = load_model('my_model.h5')

# Prompt the user for a description
description = 'apple'#input("Enter a description: ")

# Preprocess the input data
X_new = [description]

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_new)

X_new = tokenizer.texts_to_sequences(X_new)

vocab_size = len(tokenizer.word_index) + 1
maxlen = 100

X_new = pad_sequences(X_new, padding='post', maxlen=maxlen)

# Make a prediction on the new data
y_pred = model.predict(X_new)

# Print the predicted class for the item
predicted_class = np.argmax(y_pred[0])
print(f"Predicted class ID is {predicted_class}")
