import numpy as np
import pandas as pd

#Restaurant Recommendation System. Uses Pearson Correlation Coefficient
class CorrelationSystem:
    def __init__(self):
        self.recommendations = [] #Empty set at construction

    def __repr__(self):
        return "Correlation Based Restaurant Recommendation System. Uses Pearson Correlation."

    def generate_matrix(self):
        """
        @output: null, but saves correlation matrix to memory
        """
        self.frame = pd.read_csv('data/rating_final.csv')
        self.cuisine = pd.read_csv('data/chefmozcuisine.csv')
        self.geodata = pd.read_csv('data/geoplaces2.csv', encoding = 'latin1')
        #Generate means and counts for ratings
        self.rating = pd.DataFrame(self.frame.groupby('placeID')['rating'].mean())
        self.rating['rating_count'] = pd.DataFrame(self.frame.groupby('placeID')['rating'].count())
        #Pivot table of each user's rating for each place
        self.places_crosstab = pd.pivot_table(data=self.frame, values='rating', index='userID', columns='placeID')
        return self.places_crosstab

    def generate_recommendations(self, name="Restaurante Tiberius", n=5):
        """
        @param: name, string, name of restaurant to base recommendations off of
            - on a website, may be the restaurant you're currently viewing
        @param: n, int, # of recommendations to be generated
        @output: a pandas DataFrame with n rows of recommendations
            - ordered in decreasing strength of correlation
        """
        #load data\
        pid = int(self.geodata[self.geodata['name'] == name]['placeID'])
        relevant_ratings = self.places_crosstab[pid]
        #Generate pairwise correlations between those who rated current
        #restaurant and those who previously rated this restaurant.
        similar_table = self.places_crosstab.corrwith(relevant_ratings)
        correlations = pd.DataFrame(similar_table, columns=['PearsonR'])
        correlations.dropna(inplace=True)
        #
        corr_summary = correlations.join(self.rating['rating_count'])
        relevant_table = corr_summary[corr_summary['rating_count']>=10].sort_values('PearsonR', ascending=False)
        summary = pd.merge(relevant_table, self.cuisine,on='placeID')
        recommendations = pd.merge(summary, self.geodata, on='placeID')[['name', 'address', 'zip',
                     'alcohol',
                     'smoking_area',
                     'dress_code',
                     'accessibility',
                     'price',
                     'url', 'PearsonR', 'placeID']].drop_duplicates()
        return recommendations[recommendations['name'] != name].head(n)
