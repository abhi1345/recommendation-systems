import numpy as np
import pandas as pd
from models.SVD import *

class ModelSystem:

    def __init__(self):
        self.name = "Model Based Movie Recommendation System. Uses Singular Value Decomposition to Analyze Latent Factors"
        self.trained = False

    def __repr__(self):
        return self.name + " Trained: {}".format(self.trained)

    def train(self):
        #Load Data
        columns = ['user_id', 'item_id', 'rating', 'timestamp']
        frame = pd.read_csv('data/ml-100k/u.data', sep='\t', names=columns)
        columns = ['item_id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
                  'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                  'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        movies = pd.read_csv('data/ml-100k/u.item', sep='|', names=columns, encoding='latin-1')
        movie_names = movies[['item_id', 'movie title']]
        self.combined_movies_data = pd.merge(frame, movie_names, on='item_id')
        rating_crosstab = self.combined_movies_data.pivot_table(values='rating', index='user_id', columns='movie title', fill_value=0)
        X = rating_crosstab.T
        resultant_matrix = svd_fit(X, 12) #Low Rank Approximation (Cheaper Computation)
        self.corr_mat = np.corrcoef(resultant_matrix)
        self.movie_names = rating_crosstab.columns
        self.movies_list = list(self.movie_names)
        self.trained = True
        return self

    def generate_recommendations(self, name="Die Hard (1988)", n=5):
        if not self.trained: #Training once avoids unnecessary computation
            self.train()
        relevant_index = self.movies_list.index(name)
        relevant_correlations = np.round(self.corr_mat[relevant_index], 3)
        ans = []
        corr = 0.999
        i = n
        while i >= 0 and len(ans) < n:
            i_corr = list(self.movie_names[(relevant_correlations == round(i, 3))])
            if name in i_corr:
                i_corr.remove(name)
            ans.extend(i_corr)
            i -= 0.001
        return ans[:n]
