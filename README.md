# Audible Insights - Book Recommendation System

# Description
The Intelligent Book Recommendation System is a machine learning-based application designed to suggest books to users based on their preferences and interests. The system analyzes a large dataset of audiobooks by combining metadata such as genre, author, rating, reviews, price, and textual descriptions to generate personalized recommendations.

The project uses Natural Language Processing (NLP) techniques, where book descriptions and genres are combined into a single text feature and converted into numerical form using TF-IDF (Term Frequency–Inverse Document Frequency). A cosine similarity algorithm is then applied to measure the similarity between books and recommend the most relevant ones.

The system supports two types of recommendations:

Content-based recommendation – based on a selected book name.
Text-based recommendation – based on user-written description of a book.

A simple and interactive web interface is built using Streamlit, allowing users to easily search for books and receive real-time recommendations.

# Key Features
* Data cleaning and preprocessing of audiobook dataset
* Exploratory Data Analysis (EDA) for insights on ratings, genres, and authors
* NLP-based feature engineering using book descriptions and genres
* TF-IDF vectorization for text representation
* Cosine similarity for measuring book similarity
* Dual recommendation system (book name + user description)
* Interactive Streamlit web application
