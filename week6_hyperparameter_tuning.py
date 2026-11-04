
import joblib
import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Week 2 data
data = joblib.load("week2_data.pkl")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

print("Dataset loaded successfully!")

# 2. Define models and their hyperparameters
models = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {
            "C": [0.1, 1, 10]
        }
    },

    "Naive Bayes": {
        "model": MultinomialNB(),
        "params": {
            "alpha": [0.1, 0.5, 1.0]
        }
    },

    "SVM": {
        "model": LinearSVC(),
        "params": {
            "C": [0.1, 1, 10]
        }
    }
}

# 3. Cross-validation setup
cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

results = []

best_model = None
best_score = -1
best_name = ""

# 4. Perform GridSearchCV
for name, details in models.items():

    print("\nTuning:", name)

    grid = GridSearchCV(
        estimator=details["model"],
        param_grid=details["params"],
        cv=cv,
        scoring="accuracy",
        n_jobs=1
    )

    grid.fit(X_train, y_train)

    print("Best Parameters:", grid.best_params_)
    print("Best CV Accuracy:", round(grid.best_score_ * 100, 2), "%")

    # Save results
    results.append({
        "Model": name,
        "Best Parameters": str(grid.best_params_),
        "CV Accuracy": grid.best_score_
    })

    if grid.best_score_ > best_score:
        best_score = grid.best_score_
        best_model = grid.best_estimator_
        best_name = name

# 5. Evaluate best tuned model on test data
predictions = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, predictions)

print("\n========== BEST TUNED MODEL ==========")
print("Model:", best_name)
print("Cross-validation Accuracy:", round(best_score * 100, 2), "%")
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# 6. Save comparison results
results_df = pd.DataFrame(results)

print("\n========== TUNING RESULTS ==========")
print(results_df.to_string(index=False))

results_df.to_csv(
    "week6_tuning_results.csv",
    index=False
)

# 7. Save the best tuned model
joblib.dump(
    best_model,
    "best_tuned_sentiment_model.pkl"
)

print("\nBest tuned model saved successfully!")
 
