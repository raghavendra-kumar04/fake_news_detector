import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np

#Combining Dataset

# Path to the data folder
data_path = '../data'

# Read the CSV files
fake = pd.read_csv(os.path.join(data_path, 'Fake.csv'))
true = pd.read_csv(os.path.join(data_path, 'True.csv'))

# Add labels
fake['label'] = 0
true['label'] = 1

# Combine
df = pd.concat([fake, true], ignore_index=True)

# Save train.csv in the same folder
output_path = os.path.join(data_path, 'train.csv')
df[['title', 'text', 'label']].to_csv(output_path, index=False)

print("train.csv saved to:", output_path)



# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Preprocessing function
def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized)

# Load dataset
dataset_path = os.path.join(data_path, 'train.csv')
try:
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Dataset not found at {dataset_path}. Ensure train.csv is in data/.")


# Verify columns
expected_columns = ['title', 'text', 'label']
if not all(col in df.columns for col in expected_columns):
    raise ValueError("Dataset must have 'title', 'text', and 'label' columns.")

# Combine title and text
df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

# Verify labels
if df['label'].isnull().any():
    raise ValueError("Label column contains missing values.")
if not df['label'].isin([0, 1]).all():
    raise ValueError("Label column must contain only 0 (fake) or 1 (real).")

# Print label distribution
print("Label distribution:\n", df['label'].value_counts())

# Preprocess content
df['content'] = df['content'].apply(preprocess_text)

# Features and labels
X = df['content']
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Fake', 'Real'])
conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)
print("Confusion Matrix (Fake, Real):\n", conf_matrix)

# Feature importance
feature_names = vectorizer.get_feature_names_out()
importances = model.feature_importances_
top_indices = np.argsort(importances)[::-1][:10]
print("Top features:", [(feature_names[i], importances[i]) for i in top_indices])

# Save model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer saved successfully.")