import pandas as pd
import pickle

df = pd.read_csv("data/processed_data.csv")
similarity = pickle.load(open("models/similarity_matrix.pkl", "rb"))

def recommend(book_name):
    if book_name not in df['Book Name'].values:
        return ["Book not found"]

    idx = df[df['Book Name'] == book_name].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i in scores[1:10]:
        recommendations.append(df.iloc[i[0]]['Book Name'])

    return recommendations