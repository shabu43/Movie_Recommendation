import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

credits_df = pd.read_csv("/Users/shabu/Desktop/Movie_Recommendation/databases/credits.csv")

movies_df = pd.read_csv("/Users/shabu/Desktop/Movie_Recommendation/databases/movies.csv")

#printing details of both datasets
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
