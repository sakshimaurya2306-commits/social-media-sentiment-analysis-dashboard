import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from preprocess import clean_text

# Load data
df = pd.read_csv('data/dataset.csv')

# Clean text
df['text'] = df['text'].apply(clean_text)

# Split
X = df['text']
y = df['sentiment']

# Vectorize
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train
model = LogisticRegression()
model.fit(X_vec, y)

# Predict (on same data for demo)
y_pred = model.predict(X_vec)

# Confusion Matrix
cm = confusion_matrix(y, y_pred)

# Save image
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("outputs/confusion_matrix.png")
plt.close()

# Save report
report = classification_report(y, y_pred)
with open("outputs/report.txt", "w") as f:
    f.write(report)
   
# report_dict = classification_report(y_test, y_pred, output_dict=True)
# report_df = pd.DataFrame(report_dict).transpose()

# report_df.to_csv("outputs/report.csv")
# print("✅ Report saved!")

# Save model
pickle.dump(model, open('models/model.pkl', 'wb'))
pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))

print("Model trained + confusion matrix saved!")