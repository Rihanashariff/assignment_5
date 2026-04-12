import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.recommend import recommend
import pandas as pd

df = pd.read_csv("data/processed_data.csv")

st.title("📚 Audible Book Recommendation System")

book_list = df['Book Name'].values
selected_book = st.selectbox("Select a book", book_list)

if st.button("Recommend"):
    results = recommend(selected_book)
    
    st.subheader("Recommended Books:")
    for book in results:
        st.write("👉", book)