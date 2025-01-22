from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import os
import pandas as pd

# Load the model and vectorizer
with open("log_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# In-memory feedback dataset
feedback_data = []

# Test route to check server health
@app.route('/')
def home():
    return "Flask server is running!"

# Prediction endpoint
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
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

# File to store feedback data
FEEDBACK_FILE = "feedback_data.csv"

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.get_json()
        input_text = data.get('text', '')
        actual_label = data.get('actual_label', None)

        if not input_text or actual_label not in [0, 1]:
            return jsonify({"error": "Invalid feedback data"}), 400

        # Append feedback to a CSV file
        feedback_exists = os.path.isfile(FEEDBACK_FILE)
        with open(FEEDBACK_FILE, 'a') as f:
            if not feedback_exists:
                f.write("text,actual_label\n")  # Write header if file doesn't exist
            f.write(f"\"{input_text}\",{actual_label}\n")

        return jsonify({"message": "Feedback received and stored successfully."}), 200

    except Exception as e:
        app.logger.error(f"Error in feedback endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
