
# WEEK 4: DATA VISUALIZATION AND EDA

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from collections import Counter
from wordcloud import WordCloud
from nltk.corpus import stopwords

# 1. Load dataset
df = pd.read_csv("sentiment_dataset.csv")

print("DATASET LOADED SUCCESSFULLY")
print(df.head())

# 2. Check sentiment distribution
print("\nSENTIMENT DISTRIBUTION")

sentiment_counts = df["sentiment"].value_counts()

print(sentiment_counts)

# 3. Bar chart - Sentiment Distribution
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="sentiment",
    order=["positive", "negative", "neutral"]
)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig("sentiment_bar_chart.png")
plt.show()

# 4. Pie chart - Sentiment Percentage
plt.figure(figsize=(7, 7))

plt.pie(
    sentiment_counts,
    labels=sentiment_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Sentiment Percentage Distribution")
plt.tight_layout()
plt.savefig("sentiment_pie_chart.png")
plt.show()

# 5. Calculate review length
df["review_length"] = df["text"].astype(str).apply(len)

print("\nREVIEW LENGTH")
print(df[["text", "review_length"]].head())

# 6. Review length comparison
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="sentiment",
    y="review_length",
    order=["positive", "negative", "neutral"]
)

plt.title("Review Length by Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Number of Characters")
plt.tight_layout()
plt.savefig("review_length_chart.png")
plt.show()

# 7. Prepare text for word frequency analysis
stop_words = set(stopwords.words("english"))

# Preserve important negative words
stop_words.difference_update({"not", "no", "nor"})

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return words

# 8. Calculate word frequencies
all_words = []

for text in df["text"]:
    all_words.extend(clean_text(text))

word_counts = Counter(all_words)

print("\nTOP 10 MOST FREQUENT WORDS")
print(word_counts.most_common(10))

# 9. Plot most frequent words
top_words = word_counts.most_common(10)

words = [item[0] for item in top_words]
frequencies = [item[1] for item in top_words]

plt.figure(figsize=(10, 5))

sns.barplot(
    x=frequencies,
    y=words
)

plt.title("Top 10 Most Frequent Words")
plt.xlabel("Frequency")
plt.ylabel("Words")
plt.tight_layout()
plt.savefig("word_frequency_chart.png")
plt.show()

# 10. Generate overall word cloud
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white",
    colormap="viridis"
).generate(" ".join(all_words))

plt.figure(figsize=(12, 6))

plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")

plt.title("Overall Sentiment Word Cloud")
plt.tight_layout()
plt.savefig("overall_wordcloud.png")
plt.show()

# 11. Generate separate word clouds
for sentiment in ["positive", "negative", "neutral"]:

    sentiment_text = " ".join(
        df[df["sentiment"] == sentiment]["text"]
    )

    words = clean_text(sentiment_text)

    if words:
        cloud = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate(" ".join(words))

        plt.figure(figsize=(10, 5))

        plt.imshow(cloud, interpolation="bilinear")
        plt.axis("off")

        plt.title(sentiment.upper() + " SENTIMENT WORD CLOUD")
        plt.tight_layout()

        plt.savefig(sentiment + "_wordcloud.png")
        plt.show()

# 12. Display average review length
print("\nAVERAGE REVIEW LENGTH BY SENTIMENT")

print(
    df.groupby("sentiment")["review_length"].mean()
)

# 13. Save analysis dataset
df.to_csv("visualization_dataset.csv", index=False)

print("\nWEEK 4 COMPLETED SUCCESSFULLY!")
print("All graphs and word clouds generated.")
