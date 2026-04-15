import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import sys
import os

# Add project root to path to import nlp.preprocess
sys.path.append(os.getcwd())
from nlp.preprocess import clean_text

def train():
    print("Loading dataset...")
    df = pd.read_csv("data/drug_dataset.csv")
    
    print("Preprocessing symptoms...")
    df['cleaned_symptoms'] = df['symptoms'].apply(clean_text)
    
    X = df['cleaned_symptoms']
    y = df['condition']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    print("Training Logistic Regression Model...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_tfidf, y_train)
    
    # Evaluation
    y_pred = clf.predict(X_test_tfidf)
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save models
    print("Saving models to models/ directory...")
    joblib.dump(clf, "models/drug_classifier.pkl")
    joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
    print("Training complete.")

if __name__ == "__main__":
    train()
