from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load model and vectorizer
try:
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except FileNotFoundError:
    print("Error: model.pkl or vectorizer.pkl not found. Run model.py first.")
    exit(1)

# Preprocessing function
def preprocess_text(text, is_title=False):
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty or whitespace-only.")
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    # Use minimal stopwords for titles to retain context
    stop_words = set(stopwords.words('english')) if not is_title else set(['the', 'a', 'an', 'and'])
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    processed = ' '.join(lemmatized)
    if not processed:
        raise ValueError("Processed text is empty after preprocessing.")
    return processed

@app.route('/')
def home():
    return jsonify({'message': 'Fake News Detector API is running. Use POST /predict for predictions.'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"Received request: {data}")  # Debug log to console
        if not data or 'text' not in data or not data['text'].strip():
            print("Error: Invalid input - text missing or empty")
            return jsonify({'error': 'Input text is required and cannot be empty.'}), 400
        
        # Check if input is a title
        is_title = data.get('is_title', False)
        
        # Preprocess input
        processed_text = preprocess_text(data['text'], is_title=is_title)
        
        # Use title-specific vectorizer for short inputs
        if is_title:
            title_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            title_vectorizer.fit([processed_text] + vectorizer.get_feature_names_out().tolist()[:5000])
            vec_text = title_vectorizer.transform([processed_text])
        else:
            vec_text = vectorizer.transform([processed_text])
        
        # Predict
        prediction = model.predict(vec_text)[0]
        proba = model.predict_proba(vec_text)[0]
        confidence = max(proba)
        result = 'Real' if prediction == 1 else 'Fake'
        
        print(f"Prediction: {result}, Confidence: {confidence}")  # Debug log
        return jsonify({'result': result, 'confidence': f"{confidence:.2f}"})
    
    except ValueError as ve:
        print(f"ValueError: {str(ve)}")  # Debug log
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        print(f"Unexpected error: {str(e)}")  # Debug log
        return jsonify({'error': 'An unexpected error occurred.'}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)