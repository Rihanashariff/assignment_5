import streamlit as st
import pandas as pd
import pickle
from model import recommend, recommend_new_book


# ---------------- UI ---------------- #
st.title("📚 Book Recommendation")

# ---------------- LOAD DATA ---------------- #
df = pd.read_csv("data/cleaned_audible_data.csv")

df["clean_name"] = df["book name"].str.lower()
df["content"] = df["ranks and genre"] + " " + df["description"]

# ---------------- LOAD MODEL ---------------- #
tfidf = pickle.load(open("tfidf.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))


# ---------------- INPUT TYPE ---------------- #
option = st.radio("Choose Input Type", ["Book Name", "Describe Book"])

if option == "Book Name":
    book_list = sorted(df["book name"].dropna().unique())
    user_input = st.selectbox("🔎 Select or search Book Name", book_list)
else:
    user_input = st.text_area("✍ Describe the Book (genre, story, keywords)")




# ---------------- BUTTON ---------------- #
if st.button("Recommend"):
    
    if option == "Book Name":
        result = recommend(user_input, df, similarity)

        if result is None:
            st.error("❌ Book not found.")
        else:
            st.success("📖 Recommendations")
            st.table(result)

    else:
        result = recommend_new_book(user_input, df, tfidf)

        if result is None:
            st.error("❌ Please enter description.")
        else:
            st.success("📖 Recommended Books")
            for book in result:
                st.write("👉", book)