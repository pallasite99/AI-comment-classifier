from flask import Flask, request, render_template, jsonify
import pickle
import re

app = Flask(__name__)

# Load the trained model and vectorizer
with open("log_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Preprocessing function
def clean_text_no_nltk(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = {'the', 'and', 'is', 'in', 'to', 'of', 'for', 'it', 'on', 'this', 'that', 'with', 'a', 'as'}
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Route to serve the HTML page
@app.route("/")
def home():
    return render_template("index.html")  # Serves the HTML file

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.json.get("text", "")
    if not input_text:
        return jsonify({"error": "No input text provided"}), 400

    # Preprocess the input text
    cleaned_text = clean_text_no_nltk(input_text)

    # Transform text with vectorizer and make prediction
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    prediction_text = "AI-generated" if prediction == 1 else "Human-written"

    return jsonify({"prediction": prediction_text})

if __name__ == "__main__":
    app.run(debug=True)
