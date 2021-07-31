from re import I
from numpy.core.fromnumeric import sort
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, paired_distances
import pandas as pd
import numpy as np

df = pd.read_csv("final.csv")
df = df[df['soup'].notna()]

count = CountVectorizer(stop_words='English')
count_matrix = count.fit_transform(df['soup'])
cosine_sim = cosine_similarity(count_matrix,count_matrix)
df = df.reset_index()
indices = pd.Series(df.index,index=df['title'])

def getrecommendations(title):
    idx = indices[title]
    sim_score = list(enumerate(cosine_sim[idx]))
    sim_score = sorted(sim_score,key = lambda x:x[1],reverse=True)
    sim_score = sim_score[1:11]
    movie_indices = [i[0] for i in sim_score]
    return df[['title','poster_link','release_date','runtime','vote_average','overview']].iloc(movie_indices).values.tolist()