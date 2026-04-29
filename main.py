import pandas as pd
from src.text_cleaning import clean_text
from src.preprocessing import preprocess
from src.feature_extraction import get_tfidf_features
from src.model_training import train_model, evaluate_model
import joblib

df = pd.read_csv("data/social_media.csv")
df['clean'] = df['text'].apply(clean_text).apply(preprocess)

X, vectorizer = get_tfidf_features(df['clean'])
y = df['label']

joblib.dump(vectorizer, "models/vectorizer.pkl")

model = train_model(X, y)
acc, cm = evaluate_model(model, X, y)

print("Training Accuracy:", acc)
print("Confusion Matrix:\n", cm)
