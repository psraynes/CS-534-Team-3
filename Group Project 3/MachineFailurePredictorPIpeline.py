import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier,RandomForestClassifier

from data_processing import *


def main():
    data, samp_features, samp_class = load_and_sample('ai4i2020.csv', 'Machine failure')
    num_obs, num_features = samp_features.shape

    # todo: train test split 70/30

    # multi-layer perceptron. Tuning will be on hidden_layer_sizes and activation function:
    # hidden_layer_sizes, activation
    mlp = MLPClassifier()
    # Define tuning net
    # todo: Establish tuning schedule for hidden layers or try random search.
    # looking for a good way to tune the hidden layers.
    mlp_grid = {'hidden_layer_sizes': [],
                'activation': ['identity', 'logistic', 'tanh', 'relu']
                }

    # Support Vector Machine. Tuning on the regularization parameter and kernel function:
    # C, kernel
    svm = SVC()
    # todo: set tuning grids for svm
    svm_grid = {'C': [],
                'kernel': []
                }

    # Bagging Classifier with Decision Tree Classifier used. Tune the tree count and observations:
    # n_estimators, max_samples
    bag = BaggingClassifier()
    # todo: set tuning grid for bag classifier
    bag_grid = {'n_estimators': [],
                'max_samples': []
                }

    # Ada Boost Classifier with Decision Trees. Tune tree count and learning rate:
    # n_estimators, learning_rate
    ada = AdaBoostClassifier()
    # todo: set tuning grids for Adaboost
    ada_grid = {'n_estimators': [],
                'learning_rate': []
                }

    # Random Forest Classifier with Decision Trees. Tune tree count, splitting criterion, features used, tree depth, and
    # observations used in each tree:
    # n_estimators, criterion, max_features, max_depth, max_samples
    rdf = RandomForestClassifier()
    rdf_grid = {'n_estimators': [l for l in range(10, 100, 10)]+[k for k in range(100, 1001, 50)],
                'criterion': ['gini', 'entropy', 'log_loss'],
                'max_features': [i for i in range(2, num_features, 1)],
                'max_depth': [j for j in range(2, int(num_obs**.5), 1)],
                'max_samples': [n for n in np.linspace(0, 1, 11)]
                }

    print(data.shape)
    

if __name__ == "__main__":
    main()
