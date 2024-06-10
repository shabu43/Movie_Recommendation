import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

movies_df = pd.read_csv("/Users/shabu/Desktop/Movie_Recommendation/databases/pre_processing_done.csv")

CountVe = CountVectorizer(max_features=5000, stop_words='english')
v = CountVe.fit_transform(movies_df["tags"]).toarray()
