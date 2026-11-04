
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# 1. Load the training and testing data from Week 2
data = joblib.load("week2_data.pkl")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

print("Dataset loaded successfully!")

# 2. Create four machine learning models
models = {
    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "SVM": LinearSVC(),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    )
}

# 3. Train and evaluate each model
results = []

best_model = None
best_accuracy = -1
best_model_name = ""

for name, model in models.items():

    print("\nTraining:", name)

    # Train the model
    model.fit(X_train, y_train)

    # Predict test data
    predictions = model.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    )

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision, 2))
    print("Recall:", round(recall, 2))
    print("F1 Score:", round(f1, 2))

    # Select the model with the highest accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# 4. Display the comparison table
results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# 5. Save comparison results
results_df.to_csv(
    "week5_model_comparison.csv",
    index=False
)

# 6. Plot model accuracy comparison
plt.figure(figsize=(9, 5))

plt.bar(
    results_df["Model"],
    results_df["Accuracy"]
)

plt.title("Machine Learning Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig("week5_model_comparison.png")
plt.show()

# 7. Save the best model
joblib.dump(best_model, "best_sentiment_model.pkl")

print("\nBest Model:", best_model_name)
print("Best Accuracy:", round(best_accuracy * 100, 2), "%")

print("\nBest model saved successfully!")
 
