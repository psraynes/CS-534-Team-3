
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

path = "C:/Users/Owner/Desktop/processed_data/CCSC512Train.csv"

df = pd.read_csv(path)

down_sampled_df = df.sample(n=df.shape[0]//512, random_state=534)

print("Finally loaded the data!")

features = down_sampled_df[["h","s","v","con1","cor1","con2","cor2","con3","cor3","con4","cor4"]]
labels = down_sampled_df['label']

clf_knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
clf_mlp = MLPClassifier(hidden_layer_sizes=(100, 50, 25, ))

mlp_grid = {
            'activation': ['identity', 'logistic', 'tanh', 'relu']
            }

mlp_search = GridSearchCV(clf_mlp, mlp_grid, scoring='f1', n_jobs=-1, cv=5)
mlp_search.fit(features, labels)
mlp_params = mlp_search.best_params_
mlp_score = mlp_search.score(features, labels)

clf_knn.fit(features, labels)

print("Nearest Neighbors Training Accuracy:", clf_knn.score(features, labels))

print("Multi-Layer Perceptron Training Accuracy:", mlp_score)

