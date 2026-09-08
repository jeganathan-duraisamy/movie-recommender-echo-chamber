import pandas as pd
from surprise import SVD, Dataset, Reader
import pickle

ratings = pd.read_csv("/Users/Jegan/Documents/01-Project/ml-25m/ratings.csv")
print("Dataset Loaded")

reader = Reader(rating_scale=(0.5, 5))
print("Reader Completed")
data_surprise = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
print("data sprise completed")

trainset = data_surprise.build_full_trainset()
print("trainset completed")
svd_model = SVD()
svd_model.fit(trainset)
print("Model fitting completed")

with open('svd_model.pkl', 'wb') as model_file:
    pickle.dump(svd_model, model_file)

print("SVD model trained and pickled successfully!")
