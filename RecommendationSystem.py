import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#fetch the movielens dataset
data = fetch_movielens(min_rating=4.0)

#displaying training and testing data
print(repr(data['train']))
print(repr(data['test']))
