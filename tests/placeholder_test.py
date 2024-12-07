import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)

# Create mock data
mock_data = {
    'text': [
        "This is an example of human-written text.",
        "The AI generated this piece of text for testing.",
        "Humans write text like this for documentation purposes.",
        "AI can create very convincing human-like text examples.",
        "This is another piece of human-written content.",
        "AI has been used to generate this sample text."
    ],
    'generated': [0, 1, 0, 1, 0, 1]  # 0 for human, 1 for AI
}

# Convert mock data to a DataFrame
df = pd.DataFrame(mock_data)

# Total entries
print("Total entries:", df.count().sum())

# A brief overview
print(df.describe())

# Class distribution in 'generated' column
class_distribution = df['generated'].value_counts()
print("\nClass Distribution:\n", class_distribution)

# Function to clean text without NLTK
def clean_text_no_nltk(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = {'the', 'and', 'is', 'in', 'to', 'of', 'for', 'it', 'on', 'this', 'that', 'with', 'a', 'as'}
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


# Apply the cleaning function to the 'text' column
df['cleaned_text'] = df['text'].apply(clean_text_no_nltk)

# Display the first few rows to verify cleaning
print(df[['text', 'cleaned_text']].head())

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['generated']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Logistic Regression
log_model = LogisticRegression(max_iter=1000, solver='liblinear')
log_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = log_model.predict(X_test)
y_score = log_model.decision_function(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Visualization: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Visualization: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# Visualization: Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_score)
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, marker='.', label='Precision-Recall')
plt.fill_between(recall, precision, alpha=0.3, color='blue')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

# Top Positive and Negative Features
feature_names = vectorizer.get_feature_names_out()
coefficients = log_model.coef_[0]

coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
top_positive = coef_df.nlargest(10, 'Coefficient')
top_negative = coef_df.nsmallest(10, 'Coefficient')

# Plot top positive coefficients
plt.figure(figsize=(8, 5))
sns.barplot(x='Coefficient', y='Feature', data=top_positive, palette='Greens')
plt.title('Top 10 Positive Features (AI Indicating Words)')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.show()

# Plot top negative coefficients
plt.figure(figsize=(8, 5))
sns.barplot(x='Coefficient', y='Feature', data=top_negative, palette='Reds')
plt.title('Top 10 Negative Features (Human Indicating Words)')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.show()

# Visualization: Histogram of Predictions
plt.figure(figsize=(6, 4))
sns.histplot(y_score, kde=True, bins=30, color='purple')
plt.axvline(0, color='red', linestyle='--', label='Decision Boundary')
plt.title('Histogram of Decision Function Scores')
plt.xlabel('Decision Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()
