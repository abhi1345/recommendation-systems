import pandas as pd
import numpy as np

#Restaurant Recommendation System. Popularity-based.
class PopularitySystem:
    def __init__(self):
        self.output = False
    def get_data(self):
        self.frame = pd.read_csv('data/rating_final.csv')
        self.cuisine = pd.read_csv('data/chefmozcuisine.csv')
        self.geodata = pd.read_csv('data/geoplaces2.csv', encoding = 'latin1')
    def generate_recommendations(self, n=5):
        if not isinstance(self.output, pd.DataFrame):
            #Gets counts of ratings, sorts df to get most frequently rated restaurants
            rating_count = pd.DataFrame(self.frame.groupby('placeID')['rating'].count()).sort_values('rating', ascending=False)
            rating_count['placeID'] = rating_count.index
            rating_count = rating_count.drop_duplicates(subset='placeID')
            #Gets sorted placeID values, to be used for join (pandas merge) operation with geodata
            topn = rating_count.sort_values('rating', ascending=False)['rating'].index.values.astype(int)
            #Checks for out-of-bounds input
            if n > len(topn) or n < 1:
                return "Too Large/Small. Choose number between 1 and {}.".format(len(topn) - 1)
            most_rated_places = pd.DataFrame(topn, index=np.arange(len(topn)), columns=['placeID'])
            summary = pd.merge(most_rated_places, self.cuisine, on='placeID')
            #Saves sorted recommendations for future use
            self.output = pd.merge(summary, self.geodata, on='placeID')[['name', 'address', 'zip',
             'alcohol',
             'smoking_area',
             'dress_code',
             'accessibility',
             'price',
             'url', 'placeID']].drop_duplicates(subset='placeID')
        return self.output.head(n)
