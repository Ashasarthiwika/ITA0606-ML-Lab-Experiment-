
# Smart Email Spam Detection Using Naive Bayes

# Import required libraries
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load dataset
print("Loading dataset...")

data = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    names=["label", "message"]
)

# Step 2: Check dataset
print("Dataset loaded successfully!")
print(data.head())

# Step 3: Convert labels into numbers
data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

# Step 4: Split input and output
X = data["message"]
y = data["label"]

# Step 5: Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 6: Create Naive Bayes pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", MultinomialNB())
])

# Step 7: Train the model
print("\nTraining Naive Bayes model...")
model.fit(X_train, y_train)

# Step 8: Test the model
predictions = model.predict(X_test)

# Step 9: Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Training Completed!")
print("Accuracy:", round(accuracy * 100, 2), "%")

# Step 10: Display performance
print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Legitimate", "Spam"]
))

# Step 11: Save trained model
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/spam_classifier.pkl")

print("\nModel saved successfully!")
