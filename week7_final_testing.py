
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# 1. Load Week 2 test data
data = joblib.load("week2_data.pkl")

X_test = data["X_test"]
y_test = data["y_test"]

# 2. Load the best tuned model from Week 6
model = joblib.load("best_tuned_sentiment_model.pkl")

# 3. Load TF-IDF vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")

print("Model loaded successfully!")

# 4. Evaluate the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n========== FINAL MODEL EVALUATION ==========")

print("Final Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# 5. Display confusion matrix
cm = confusion_matrix(
    y_test,
    predictions,
    labels=["negative", "neutral", "positive"]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Neutral", "Positive"]
)

display.plot(cmap="Blues")
plt.title("Final Sentiment Analyzer Confusion Matrix")
plt.tight_layout()
plt.savefig("week7_confusion_matrix.png")
plt.show()

# 6. Test new sentences
new_reviews = [
    "I love this product",
    "This service is terrible",
    "The product is average",
    "Amazing quality and excellent service",
    "I am very disappointed",
    "The product is okay",
    "I am happy with my purchase",
    "The experience was very bad"
]

# 7. Convert sentences into TF-IDF features
new_features = vectorizer.transform(new_reviews)

# 8. Predict sentiments
new_predictions = model.predict(new_features)

# 9. Display predictions
print("\n========== NEW SENTENCE PREDICTIONS ==========")

for review, sentiment in zip(new_reviews, new_predictions):

    print("\nReview:", review)
    print("Predicted Sentiment:", sentiment.upper())

# 10. Save predictions
prediction_df = pd.DataFrame({
    "Review": new_reviews,
    "Predicted Sentiment": new_predictions
})

prediction_df.to_csv(
    "week7_predictions.csv",
    index=False
)

print("\nPredictions saved successfully!")
 
