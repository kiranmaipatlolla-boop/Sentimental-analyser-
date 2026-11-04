
from flask import Flask, render_template, request
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("best_tuned_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Initialize preprocessing tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# Preserve important negation words
stop_words.difference_update({"not", "no", "nor"})


# Text preprocessing function
def preprocess_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


@app.route("/", methods=["GET", "POST"])
def home():

    sentiment = None
    review = ""

    if request.method == "POST":

        review = request.form.get("review", "").strip()

        if review:

            cleaned_text = preprocess_text(review)

            text_features = vectorizer.transform(
                [cleaned_text]
            )

            prediction = model.predict(text_features)

            sentiment = prediction[0].upper()

    return render_template(
        "index.html",
        sentiment=sentiment,
        review=review
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )

