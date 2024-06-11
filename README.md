# Movie Recommendation System Project

## Project Overview

This project aims to build a movie recommendation system using Natural Language Processing (NLP) techniques and cosine similarity. The recommendation system suggests movies based on the similarity of their tags, which are derived from various features of the movies such as genres, cast, crew, keywords, and an overview. The project involves preprocessing movie data, transforming it into meaningful features, and calculating similarities to provide recommendations.

## Python packages required
pip install numpy pandas scikit-learn nltk

## Data Preprocessing
- Load the credits.csv and movies.csv datasets.
- Merge the datasets on the movie title.
- Handle missing values by dropping rows with missing overview or release_date.
- Transform the release_date to represent decades.
- Extract and transform features like genres, keywords, cast, and crew to lists of relevant names.
- Combine these features into a single tags column for each movie.
- Save the processed DataFrame to pre_processing_done.csv.
  
## Recommendation System
- Load the preprocessed dataset pre_processing_done.csv.
- Create a TF-IDF vectorizer to transform the tags into vectors.
- Apply Porter Stemmer to the tags to reduce words to their root forms.
- Calculate cosine similarity between the movie vectors.
- Define a recommend function to suggest movies based on a given title.

