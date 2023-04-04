# Flask API for OpenAI Trademark Class Recommender
This is a Flask API that uses OpenAI's GPT-3 language model to generate trademark class recommendations for a given product or service description.

## Installation
To use the API, you will need to have Python 3 installed, as well as the flask and openai Python packages. You can install these packages using pip:

`pip install flask openai`
You will also need an OpenAI API key, which you can obtain from the OpenAI website.
Place it on line 5 on `app_using_openAI_api.py`

## Usage
To use the API, you can start the Flask app using the following command:

```
python app_using_openAI_api.py
```
This will start the app and make it available at http://localhost:8001.

To receive trademark class recommendations for a given product or service description, you can send a POST request to the /recommend endpoint with a JSON payload containing a description field, like so:


```POST /recommend HTTP/1.1
Host: localhost:8001
Content-Type: application/json

{
    "description": "A device for cleaning floors"
}
```
This will generate trademark class recommendations for the given description using OpenAI's GPT-3 language model. The recommended classes will be returned as a JSON response:

```
HTTP/1.1 200 OK
Content-Type: application/json

{
    "recommended_classes": "Class 7: Machines and Machine Tools, Class 11: Environmental Control Apparatus, Class 21: Housewares and Glass, Class 35: Advertising and Business"
}
```

## Acknowledgments
This project was created using the OpenAI API, which provides access to GPT-3 language models, and the Flask web framework.