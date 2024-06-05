import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import ast


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
Mmovies_df['release_date'] = Mmovies_df.release_date.astype(str)
#print(Mmovies_df['release_date'].min)

#Mmovies_df.info()
#print(Mmovies_df['release_date'].min)

#keep only necessary collumn

Mmovies_df = Mmovies_df[["movie_id","title","overview","genres","keywords","cast","crew","release_date"]]

#transform Keywords and genres name to list of literals
def transform(obj):
    List=[]
    
    for i in ast.literal_eval(obj):
        List.append(i["name"])
    return List

Mmovies_df['genres']=Mmovies_df['genres'].apply(transform)
Mmovies_df['keywords']=Mmovies_df['keywords'].apply(transform)

#print(Mmovies_df["genres"].head())
#print(Mmovies_df["keywords"].head())

#transform cast name to list of literals, maximum 3 names.
def transform1(obj):
    List=[]
    c=0
    for i in ast.literal_eval(obj):
        if c!=3:
            List.append(i["name"])
            c=c+1
    return List
Mmovies_df['cast']=Mmovies_df['cast'].apply(transform1)
#print(Mmovies_df["cast"].head())

#transform in crew and keep only Director name as list of literal.
def transform2(obj):
    List=[]
    
    for i in ast.literal_eval(obj):
        if i["job"]=="Director":
            List.append(i["name"])
    return List
Mmovies_df['crew']=Mmovies_df['crew'].apply(transform2)
#print(Mmovies_df["crew"].head())

Mmovies_df['overview'] = Mmovies_df['overview'].apply(lambda x:x.split())
Mmovies_df['genres'] = Mmovies_df['genres'].apply(lambda x:[i.replace(" ","") for i in x])
Mmovies_df['crew'] = Mmovies_df['crew'].apply(lambda x:[i.replace(" ","") for i in x])
Mmovies_df['cast'] = Mmovies_df['cast'].apply(lambda x:[i.replace(" "," ") for i in x])
Mmovies_df['keywords'] = Mmovies_df['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
#print(Mmovies_df["overview"].head())

#transform release_date to list of literals
def transform3(obj):
    List=[]
    c=0
    for i in obj:
        if c==0:
            List.append(obj)
            c=c+1
    return List
Mmovies_df['release_date']=Mmovies_df['release_date'].apply(transform3)
#print(Mmovies_df["release_date"].head())

# Create new column 'tags' with all columns tag together
Mmovies_df['tags'] = Mmovies_df["overview"]+Mmovies_df["genres"] + Mmovies_df["keywords"]+Mmovies_df["cast"]+Mmovies_df["crew"]+Mmovies_df["release_date"]

#reduce dataframe with movie identity and tags
Mmovies_df = Mmovies_df[["movie_id","title","tags"]]
Mmovies_df['tags'] = Mmovies_df['tags'].apply(lambda x: ' '.join(x))
#print(Mmovies_df["tags"].head())

#saving the processed dataframe
print("\nSaving data...\n.\n.")
Mmovies_df.to_csv('/Users/shabu/Desktop/Movie_Recommendation/databases/pre_processing_done.csv')
print("Saved!")
