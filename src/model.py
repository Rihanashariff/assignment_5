import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# Load TF-IDF matrix
tfidf_matrix = pickle.load(open("models/tfidf_matrix.pkl", "rb"))

# Similarity
similarity = cosine_similarity(tfidf_matrix)

# Save similarity matrix
pickle.dump(similarity, open("models/similarity_matrix.pkl", "wb"))

# Clustering
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix)

pickle.dump(kmeans, open("models/clustering_model.pkl", "wb"))

print("Model training completed!")