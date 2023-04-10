import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier,RandomForestClassifier

from data_processing import *


def main():
    data, samp_features, samp_class = load_and_sample('ai4i2020.csv', 'Machine failure', ['UDI', 'Product ID'])
    num_obs, num_features = samp_features.shape
    
    # Convert L, M, H strings to 0, 1, 2 numbers
    samp_features = samp_features.applymap(encodeLMH)

    # Split data into training and testing sets
    features_train, features_test, class_train, class_test = train_test_split(samp_features, samp_class, train_size=0.7)
    
    # multi-layer perceptron. Tuning will be on hidden_layer_sizes and activation function:
    # hidden_layer_sizes, activation
    mlp = MLPClassifier()
    # Define tuning net
    # todo: Establish tuning schedule for hidden layers or try random search.
    # looking for a good way to tune the hidden layers.
    mlp_grid = {'hidden_layer_sizes': [],
                'activation': ['identity', 'logistic', 'tanh', 'relu']
                }
    
    mlp.fit(features_train, class_train)

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
