import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

movies_df = pd.read_csv("/Users/shabu/Desktop/Movie_Recommendation/databases/pre_processing_done.csv")

#creating vector with tokenize words from 'tags' created in pre-processing
CountVe = CountVectorizer(max_features=5000, stop_words='english')
v = CountVe.fit_transform(movies_df["tags"]).toarray()

#refining words to their root with porter-stemmer algo
porterS = PorterStemmer()
def stemm(text):
    x=[]
    for i in text.split():
        x.append(porterS.stem(i))
    return " ".join(x)

movies_df["tags"] = movies_df["tags"].apply(stemm)
#print(movies_df["tags"].head())

#Find similarity between each movie entity using cosine_similarity
v = CountVe.fit_transform(movies_df["tags"]).toarray()
similarity = cosine_similarity(v)
#print(sorted(list(enumerate(similarity[0])), reverse=True, key = lambda x:x[1])[1:6])

#recomend movie based on similarities
def recommend(title):
    m_index = movies_df[movies_df['title'] == title].index[0]
    distances = similarity[m_index]
    mo_list = sorted(list(enumerate(distances)), reverse = True, key = lambda x:x[1])[1:7]
    for i in mo_list:
        print(movies_df.iloc[i[0]].title)
        
val = input("Enter a movie title: ")

r=0
for i in movies_df["title"]:
    if i == val:
        recommend(val)
        r=1
        break;
if r == 0:
        print("sorry!! the movoie is Not recognized by the system!!\nPlease insert a correct Title")

