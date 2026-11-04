
# WEEK 3: LOGISTIC REGRESSION MODEL BUILDING

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Load Week 2 prepared data
data = joblib.load("week2_data.pkl")

X_train = data["X_train"]
X_test = data["X_test"]

y_train = data["y_train"]
y_test = data["y_test"]

print("WEEK 2 DATA LOADED SUCCESSFULLY")

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# 2. Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# 3. Train the model
model.fit(X_train, y_train)

print("\nMODEL TRAINED SUCCESSFULLY!")

# 4. Predict test data
y_pred = model.predict(X_test)

print("\nPREDICTED SENTIMENTS")
print(y_pred)

# 5. Display actual sentiments
print("\nACTUAL SENTIMENTS")
print(y_test.to_numpy())

# 6. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY")
print(f"{accuracy * 100:.2f}%")

# 7. Classification report
print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        y_pred,
        labels=model.classes_,
        zero_division=0
    )
)

# 8. Confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

print("\nCONFUSION MATRIX")
print(cm)

# 9. Display confusion matrix graph
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

display.plot(cmap="Blues")

plt.title("Sentiment Analysis - Confusion Matrix")
plt.show()

# 10. Save trained model
joblib.dump(model, "logistic_regression_model.pkl")

print("\nMODEL SAVED SUCCESSFULLY!")

# 11. Load TF-IDF vectorizer
tfidf = joblib.load("tfidf_vectorizer.pkl")

# 12. Test with new sentences
test_sentences = [
    "I love this product",
    "This service is terrible",
    "The product is average"
]

# 13. Convert new sentences into TF-IDF features
X_new = tfidf.transform(test_sentences)

# 14. Predict new sentiments
new_predictions = model.predict(X_new)

print("\nNEW SENTIMENT PREDICTIONS")

for sentence, sentiment in zip(test_sentences, new_predictions):
    print(f"Text: {sentence}")
    print(f"Sentiment: {sentiment.upper()}")
    print("-----------------------------")

print("\nCOMPLETED SUCCESSFULLY!")
