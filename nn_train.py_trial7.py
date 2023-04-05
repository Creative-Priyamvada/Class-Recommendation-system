import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the JSON data
with open('idmanual.json') as f:
    data = json.load(f)

# Create the input and target arrays
X = []
y = []
for item in data:
    X.append(item['description'])
    y.append(item['class_id'])

# Convert target array to integers using LabelEncoder
le = LabelEncoder()
le.fit(y)
y = le.transform(y)

# Convert target array to one-hot encoding
num_classes = len(le.classes_)
y = to_categorical(y, num_classes=num_classes)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#print(X_train)

# Preprocess the input data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1
maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# Define the model architecture
model = Sequential()
model.add(Dense(64, input_shape=(maxlen,), activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))


# Save the model to disk
model.save('my_model.h5')


# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {loss:.3f}")
print(f"Test accuracy: {accuracy:.3f}")

# Make predictions on the test data
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# Calculate precision, recall, and F1-score
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1score, _ = precision_recall_fscore_support(np.argmax(y_test, axis=1), y_pred, average='weighted')
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-score: {f1score:.3f}")