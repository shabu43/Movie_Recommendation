import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

credits_df = pd.read_csv("/Users/shabu/Desktop/Movie_Recommendation/databases/credits.csv")

movies_df = pd.read_csv("/Users/shabu/Desktop/Movie_Recommendation/databases/movies.csv")

#printing details of both datasets
"""
print("\nCredits head")
print(credits_df.head())

print("Credits tail")
print(credits_df.tail())

print("\nMovies head")
print(movies_df.head())

print("Movies tail")
print(movies_df.tail())

print("\nMovies info")
print(credits_df.info())
print(credits_df.shape)

print("\nCredits info")
print(movies_df.info())
print(movies_df.shape)
"""

#Merging/joining both dataset together based on movie Title
Mmovies_df = movies_df.merge(credits_df,on = "title")

#finding missing values in the dataset
print(Mmovies_df.isna().sum())
#dropping necessary coulumn's missing rows. [only 4 missng overviw, 1 release date]
Mmovies_df = Mmovies_df.dropna(subset=["overview","release_date"])
print(Mmovies_df.isna().sum())
print(Mmovies_df.shape)

