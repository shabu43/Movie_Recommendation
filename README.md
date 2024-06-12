# Movie Recommendation System Project

## Project Overview

This project aims to build a movie recommendation system using Natural Language Processing (NLP) techniques and cosine similarity. The recommendation system suggests movies based on the similarity of their tags, which are derived from various features of the movies such as genres, cast, crew, keywords, and an overview. The project involves preprocessing movie data, transforming it into meaningful features, and calculating similarities to provide recommendations.

## Project Files
- credits.csv: Contains information about the cast and crew of movies.
- movies.csv: Contains detailed information about movies including genres, keywords, and an overview.
- pre_processing_done.csv: The processed dataset with relevant features combined into tags for each movie.
- movie_recommendation.py: The main script for preprocessing data, calculating similarities, and recommending movies.
- README.md: Documentation file explaining the project.

## Steps to Run the Project

### Python packages required
pip install numpy pandas scikit-learn nltk

### Data Preprocessing
- Load the credits.csv and movies.csv datasets.
- Merge the datasets on the movie title.
- Handle missing values by dropping rows with missing overview or release_date.
- Transform the release_date to represent decades.
- Extract and transform features like genres, keywords, cast, and crew to lists of relevant names.
- Combine these features into a single tags column for each movie.
- Save the processed DataFrame to pre_processing_done.csv.
  
### Recommendation System
- Load the preprocessed dataset pre_processing_done.csv.
- Create a TF-IDF vectorizer to transform the tags into vectors.
- Apply Porter Stemmer to the tags to reduce words to their root forms.
- Calculate cosine similarity between the movie vectors.
- Define a recommend function to suggest movies based on a given title.

To get movie recommendations, run the script (movie_recommendation.py) and input a movie title when prompted: "Enter a movie title:"

## Conclusion

This movie recommendation system leverages NLP and cosine similarity to suggest movies based on their similarity to a given movie. The preprocessing steps ensure that the data is cleaned and transformed into a format suitable for similarity calculations. By applying stemming and vectorization, the system can efficiently compute similarities and provide accurate recommendations.
