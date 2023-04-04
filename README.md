# Class Recommendation System
This is a Flask-based API for a class recommendation system. The system takes in a description of a product or service and recommends classes that the product or service may belong to.


## Installation
To use the API, you will need to have Python 3 installed. You can then install the required Python packages by running:

`pip install -r requirements.txt`

## Usage

`python train1.py`

To run the API, you can use the following command:

`python app.py`

This will start the API server on port 8001. You can then send a POST request to the /recommend endpoint with a JSON payload containing a description field with the product or service description. The API will respond with a JSON payload containing a list of recommended classes and their corresponding cosine similarity scores.

Here's an example of how to use the API with Python:

```
import requests
url = 'http://localhost:8001/recommend'
data = {'description': 'Computer hardware, namely, electronic components'}
response = requests.post(url, json=data)
recommendations = response.json()['recommendations']
for recommendation in recommendations:
    print(f'Class ID: {recommendation[0]}, Probability: {recommendation[1]:.2%}')
```








