import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the CSV file
df = pd.read_csv("virus.csv")
# Combine 'covid' and 'fever' columns as input features
data = df["covid"].astype(str) + " " + df["fever"].astype(str)
X = data
y = df['flu']  # Labels

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into bag-of-words
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train classifier
classifier = MultinomialNB()
classifier.fit(X_train_counts, y_train)

# Load test dataset
data1 = pd.read_csv("test1.csv")
new_data = data1["covid"].astype(str) + " " + data1["fever"].astype(str)
new_data_counts = vectorizer.transform(new_data)

# Make predictions
predictions = classifier.predict(new_data_counts)
print("Predictions on new dataset:")
print(predictions)

# Evaluate performance
y_pred_test = classifier.predict(X_test_counts)
accuracy = accuracy_score(y_test, y_pred_test)
print(f"\nAccuracy on Test Set: {accuracy:.2f}")
print("Classification Report on Test Set:")
print(classification_report(y_test, y_pred_test))

# Save predictions to CSV
data1['flu_prediction'] = predictions
data1.to_csv("test1.csv", index=False)
print("\nPredictions saved to test1.csv successfully.")
