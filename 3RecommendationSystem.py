import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#fetch the movielens dataset
data = fetch_movielens(min_rating=4.0)
print(data.keys())

#displaying training and testing data
print(repr(data['train']))
print(repr(data['test']))

#creating the model
model = LightFM(loss='warp')

#training the model
print("Training the model...")
model.fit(data['train'], epochs=30, num_threads=2)
print("Model training completed.")

# Test prediction for user 2, item 10
score = model.predict(np.array([2]), np.array([10]))[0]
print(f"Predicted score for user 2 on item 10: {score}")

#function to sample recommendations
def sample_recommendation(model, data, user_ids):
    #number of users and items in the training data
    n_users, n_items = data['train'].shape
    #generating recommendations for each user
    for user_id in user_ids:
        #items the user already likes
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        #predicting scores for all items
        scores = model.predict(user_id, np.arange(n_items))
        top_items = data['item_labels'][np.argsort(-scores)]

        print(f"User {user_id}")
        print("     Known positives:")
        for x in known_positives[:3]:
            print(f"        {x}")

        print("     Recommended:")
        #top 3 recommendations 
        for x in top_items[:3]:
            print(f"        {x}")


#sample recommendations for users 2, 100 and 250
sample_recommendation(model, data, [2, 100, 250])
