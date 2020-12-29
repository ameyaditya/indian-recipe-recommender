import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from database_operations import *

def initialise_feature_vector():
    ingredients = get_ingredient_details()
    ingredient_name = [i[1] for i in ingredients]
    vectorizer = CountVectorizer()
    vectorizer.fit(ingredient_name)
    food_details = get_food_details(ingredients=True)
    feature_vector = []
    feature_map_dict = {}
    i = 0
    for each_food in food_details:
        vector = [i[2] for i in each_food[-1]]
        v1 = vectorizer.transform(vector)
        v1 = np.array(np.sum(v1, axis = 0))
        v1 = v1.ravel()
        feature_vector.append(v1)
        feature_map_dict[i] = each_food[0]
        i += 1
    return vectorizer, feature_vector, feature_map_dict