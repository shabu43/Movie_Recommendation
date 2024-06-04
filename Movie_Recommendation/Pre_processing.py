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
#print(Mmovies_df.isna().sum())
#dropping necessary coulumn's missing rows. [only 4 missng overviw, 1 release date]
Mmovies_df = Mmovies_df.dropna(subset=["overview","release_date"])
#print(Mmovies_df.isna().sum())
#print(Mmovies_df.shape)

#Transform Release date from speicific date to decades form .
#https://stackoverflow.com/questions/71460233/how-can-i-extract-the-year-from-object-column
#change date datatype from object to datetime to extract only year then decades.
Mmovies_df['release_date'] = pd.to_datetime(Mmovies_df['release_date'], format="mixed")
Mmovies_df['release_date'] = Mmovies_df['release_date'].apply(lambda x: x.year)
Mmovies_df['release_date'] = Mmovies_df['release_date']/10
Mmovies_df['release_date'] = Mmovies_df['release_date'].astype('int')
Mmovies_df['release_date'] = Mmovies_df['release_date'].astype('object')
#Mmovies_df.info()
#print(Mmovies_df['release_date'].min)

#keep only necessary collumn

Mmovies_df = Mmovies_df[["movie_id","title","overview","genres","keywords","cast","crew","release_date"]]

import ast

def transform(obj):
    List=[]
    
    for i in ast.literal_eval(obj):
        List.append(i["name"])
    return List

Mmovies_df['genres']=Mmovies_df['genres'].apply(transform)
Mmovies_df['keywords']=Mmovies_df['keywords'].apply(transform)
print(Mmovies_df["genres"].head())
print(Mmovies_df["keywords"].head())
