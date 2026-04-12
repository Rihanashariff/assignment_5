import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

df = pd.read_csv("data/processed_data.csv")

# Fill missing values
df['Book Name'] = df['Book Name'].fillna('')
df['Author'] = df['Author'].fillna('')
df['Description'] = df['Description'].fillna('')

# Convert everything to string (extra safety)
df['Book Name'] = df['Book Name'].astype(str)
df['Author'] = df['Author'].astype(str)
df['Description'] = df['Description'].astype(str)

# Combine features
df['content'] = df['Book Name'] + " " + df['Author'] + " " + df['Description']

# FINAL safety (very important)
df['content'] = df['content'].fillna('')

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])

# Save
pickle.dump(tfidf, open("models/tfidf_vectorizer.pkl", "wb"))
pickle.dump(tfidf_matrix, open("models/tfidf_matrix.pkl", "wb"))

print("Feature engineering completed!")