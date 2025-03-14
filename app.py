import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, render_template, request

# Initialize the Flask application
app = Flask(__name__)

# Load the dataset
df = pd.read_csv("spam.csv", encoding="latin-1", on_bad_lines='skip')

# Handle missing values
df['text'] = df['text'].fillna('')

# Prepare data for training
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Convert text data to numerical format using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Define function to predict if an email is spam
def predict_email(email_text):
    email_vec = vectorizer.transform([email_text])
    prediction = model.predict(email_vec)
    return "Spam" if prediction[0] == 1 else "Not Spam"

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_results = []
    
    if request.method == 'POST':
        # Getting multiple emails from the form
        email_texts = request.form['email_texts']
        email_list = email_texts.split("\n")
        
        # Predicting spam or not spam for each email
        for email in email_list:
            prediction = predict_email(email.strip())  # strip to clean up any extra spaces
            prediction_results.append({'email': email, 'prediction': prediction})
    
    return render_template('index.html', prediction_results=prediction_results)

if __name__ == "__main__":
    app.run(debug=True)
