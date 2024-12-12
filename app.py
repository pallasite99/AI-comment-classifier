from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re

# Load the model and vectorizer
with open("log_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Test route to check server health
@app.route('/')
def home():
    return "Flask server is running!"

# Prediction endpoint (handles both POST and non-POST methods)
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        # Handle GET request with a user-friendly message
        return jsonify({"error": "This endpoint only supports POST requests. Please send a POST request."}), 405

    try:
        data = request.get_json()
        input_text = data.get('text', '')

        if not input_text:
            return jsonify({"error": "No text provided"}), 400

        # Preprocess and vectorize input text
        cleaned_text = clean_text(input_text)
        vectorized_text = vectorizer.transform([cleaned_text])

        # Predict and get probabilities
        probabilities = model.predict_proba(vectorized_text)[0]
        prediction = model.predict(vectorized_text)[0]

        # Get the confidence score for the predicted class
        confidence = max(probabilities) * 100

        # Return prediction and confidence score
        return jsonify({
            "prediction": "Human-written" if prediction == 0 else "AI-generated",
            "confidence": confidence
        })

    except Exception as e:
        app.logger.error(f"Error in prediction endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
