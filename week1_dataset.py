# WEEK 1: UNDERSTAND DATASET AND PANDAS

import pandas as pd

# 1. Load dataset
df = pd.read_csv("sentiment_dataset.csv")

# 2. Display first 5 records
print("FIRST 5 RECORDS")
print(df.head())

# 3. Display last 5 records
print("\nLAST 5 RECORDS")
print(df.tail())

# 4. Dataset information
print("\nDATASET INFORMATION")
print(df.info())

# 5. Dataset shape
print("\nDATASET SHAPE")
print(df.shape)

# 6. Column names
print("\nCOLUMN NAMES")
print(df.columns)

# 7. Check missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# 8. Check duplicate records
print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())

# 9. Remove missing values
df = df.dropna(subset=["text", "sentiment"])

# 10. Remove duplicate records
df = df.drop_duplicates()

# 11. Remove extra spaces
df["text"] = df["text"].astype(str).str.strip()

# 12. Convert sentiment labels to lowercase
df["sentiment"] = df["sentiment"].str.lower().str.strip()

# 13. Display sentiment distribution
print("\nSENTIMENT DISTRIBUTION")
print(df["sentiment"].value_counts())

# 14. Display statistical information
print("\nDATASET DESCRIPTION")
print(df.describe(include="all"))

# 15. Save cleaned dataset
df.to_csv("cleaned_sentiment.csv", index=False)

print("\nCLEANED DATASET SAVED SUCCESSFULLY!")

# 16. Final dataset shape
print("\nFINAL DATASET SHAPE")
print(df.shape)