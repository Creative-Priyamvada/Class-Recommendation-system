from flask import Flask, jsonify, request
import openai

# Set up OpenAI API key
openai.api_key = "Enter api key here"

app = Flask(__name__)

# Define a route for the API
@app.route('/recommend', methods=['POST'])
def recommend():
    # Get the input data from the request
    data = request.json

    # Get the description from the input data
    description = data['description']

    # Use the OpenAI API to generate trademark class recommendations for the input
    response = openai.Completion.create(
        engine="davinci",
        prompt=f"What trademark classes are relevant for '{description}'?",
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the recommended classes from the API response
    recommended_classes = response.choices[0].text.strip()

    # Convert the recommendations to a JSON response
    response = jsonify({'recommended_classes': recommended_classes})

    return response

if __name__ == '__main__':
    app.run(debug=True,port=8001)
