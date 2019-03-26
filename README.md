# Small-Scale Recommendation System Prototypes
Five different types of recommendation systems used by Amazon, Netflix, LinkedIn, etc.

Each script is in OOP form. The system returns recommendations in pandas DataFrame format.

## The Code
The following five types of recommendation systems are implemented, with each in a different script file:
### Popularity-based recommendation
Recommends items based on how 'popular' they are (impersonal).
### Correlation-based recommendation
Recommends items based on Pearson correlation between another from previous user ratings.
### Classification-based collaborative filter
Uses logistic regression to give personalized recommendation.
### Model-based collaborative filter
Build model from user ratings, make recommendations from model.
### Content-based recommendation
Recommends items based on feature similarity. Uses kNN.

## System API
Each recommendation system uses the following interface:
* init: generates an instance of the system.
* generate_recommendations(n): outputs a pandas DataFrame with n rows of recommendations (1 per row).

## The Models
The last 3 systems use machine learning models to aid in generating suggestion. All models are in Python, based on pseudocode written by Johnathan Shewchuk in [Concise Machine Learning.](https://people.eecs.berkeley.edu/~jrs/papers/machlearn.pdf)

## The Data
Dataset taken from UCI machine learning repository. [link](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data)

## Dependencies
numpy, scipy, pandas

Dependencies used for acceleration of calculations. All systems/models built from scratch and can be built without libraries.
