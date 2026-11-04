
# WEEK 2: TEXT PREPROCESSING AND FEATURE EXTRACTION

import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# 1. Load dataset
df = pd.read_csv("sentiment_dataset.csv")

print("DATASET LOADED SUCCESSFULLY")
print(df.head())

# 2. Check dataset
print("\nDATASET SHAPE")
print(df.shape)

# 3. Separate input and output
X = df["text"]
y = df["sentiment"]

print("\nINPUT FEATURES")
print(X.head())

print("\nOUTPUT LABELS")
print(y.head())

# 4. Download NLTK stopwords
nltk.download("stopwords", quiet=True)

# 5. Get English stopwords
stop_words = set(stopwords.words("english"))

# 6. Create stemming object
stemmer = PorterStemmer()

# 7. Define text preprocessing function
def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", "", text)

    # Tokenization
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Stemming
    words = [stemmer.stem(word) for word in words]

    # Join words
    return " ".join(words)


# 8. Apply preprocessing
df["processed_text"] = df["text"].apply(preprocess_text)

print("\nORIGINAL AND PROCESSED TEXT")

print(df[["text", "processed_text"]].head(10))

# 9. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["processed_text"],
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTRAINING DATA SHAPE")
print(X_train.shape)

print("\nTESTING DATA SHAPE")
print(X_test.shape)

# 10. Apply TF-IDF vectorization
tfidf = TfidfVectorizer()

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF TRAINING SHAPE")
print(X_train_tfidf.shape)

print("\nTF-IDF TESTING SHAPE")
print(X_test_tfidf.shape)

# 11. Display vocabulary
print("\nTF-IDF VOCABULARY")
print(tfidf.get_feature_names_out())

# 12. Display numerical representation
print("\nTF-IDF NUMERICAL VALUES")
print(X_train_tfidf.toarray()[:3])

# 13. Save processed dataset
df.to_csv("processed_sentiment.csv", index=False)

# 14. Save TF-IDF vectorizer
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

# 15. Save prepared training and testing data
joblib.dump(
    {
        "X_train": X_train_tfidf,
        "X_test": X_test_tfidf,
        "y_train": y_train,
        "y_test": y_test
    },
    "week2_data.pkl"
)

print("\n COMPLETED SUCCESSFULLY!")
print("Processed dataset saved.")
print("TF-IDF vectorizer saved.")
print("Training and testing data saved.")
