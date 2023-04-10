import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier,RandomForestClassifier\

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
    # looking for a good way to tune the hidden layers.
    mlp_grid = {'hidden_layer_sizes': [(5,), (20,), (100,),
                                       (5,5,), (20,20,), (100,100,),
                                       (5,20,), (5,100,),
                                       (20,5,), (20,100,), 
                                       (100,5,), (100,20,)],
                'activation': ['identity', 'logistic', 'tanh', 'relu']
                }
    
    mlp_search = GridSearchCV(mlp, mlp_grid, scoring='f1', n_jobs=-1, cv=5)
    mlp_search.fit(features_train, class_train)
    mlp_params = mlp_search.best_params_
    mlp_score = mlp_search.score(features_train, class_train)

    # Support Vector Machine. Tuning on the regularization parameter and kernel function:
    # C, kernel
    svm = SVC()
    svm_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
                }

    svm_search = GridSearchCV(svm, svm_grid, scoring='f1', n_jobs=-1, cv=5)
    svm_search.fit(features_train, class_train)
    svm_params = svm_search.best_params_
    svm_score = svm_search.score(features_train, class_train)
    
    # Bagging Classifier with Decision Tree Classifier used. Tune the tree count and observations:
    # n_estimators, max_samples
    bag = BaggingClassifier()
    bag_grid = {'n_estimators': [1, 10, 100],
                'max_samples': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                }

    bag_search = GridSearchCV(bag, bag_grid, scoring='f1', n_jobs=-1, cv=5)
    bag_search.fit(features_train, class_train)
    bag_params = bag_search.best_params_
    bag_score = bag_search.score(features_train, class_train)
    
    # Ada Boost Classifier with Decision Trees. Tune tree count and learning rate:
    # n_estimators, learning_rate
    ada = AdaBoostClassifier()
    ada_grid = {'n_estimators': [10, 50, 100, 500],
                'learning_rate': [0.1, 0.5, 1, 5, 10, 50]
                }

    ada_search = GridSearchCV(ada, ada_grid, scoring='f1', n_jobs=-1, cv=5)
    ada_search.fit(features_train, class_train)
    ada_params = ada_search.best_params_
    ada_score = ada_search.score(features_train, class_train)
    
    # Random Forest Classifier with Decision Trees. Tune tree count, splitting criterion, features used, tree depth, and
    # observations used in each tree:
    # n_estimators, criterion, max_features, max_depth, max_samples
    rdf = RandomForestClassifier()
    rdf_grid = {'n_estimators': [l for l in range(25, 100, 25)]+[k for k in range(100, 501, 50)],
                'criterion': ['gini', 'entropy'],
                'max_features': [i for i in range(2, num_features, 1)],
                'max_depth': [j for j in range(2, int(num_obs**.5), 1)],
                'max_samples': [n for n in np.linspace(0.1, 1, 10)]
                }

    # We perform this one as a randomized search instead of exhaustive to maintain a reasonable run time.
    rdf_search = RandomizedSearchCV(rdf, rdf_grid, n_iter=500, scoring='f1', n_jobs=-1, cv=5)
    rdf_search.fit(features_train, class_train)
    rdf_params = rdf_search.best_params_
    rdf_score = rdf_search.score(features_train, class_train)
    
    print("{:<30} {:<64} {:<24}".format('ML Trained Model','Best Parameter Values','F1 Score on Training Data'))
    print()
    print("{:<30} {:<64} {:<24}".format('Artificial Neural Networks',str(mlp_params),mlp_score))
    print("{:<30} {:<64} {:<24}".format('Support Vector Machine',str(svm_params),svm_score))
    print("{:<30} {:<64} {:<24}".format('BaggingClassifier',str(bag_params),bag_score))
    print("{:<30} {:<64} {:<24}".format('AdaBoost',str(ada_params),ada_score))
    print("{:<30} {:<64} {:<24}".format('Random Forest ',str(rdf_params),rdf_score))
    
    print()
    
    print("{:<30} {:<64} {:<24}".format('ML Trained Model','Best Parameter Values','F1 Score on Test Data'))
    print()
    print("{:<30} {:<64} {:<24}".format('Artificial Neural Networks',str(mlp_params),mlp_search.score(features_test, class_test)))
    print("{:<30} {:<64} {:<24}".format('Support Vector Machine',str(svm_params),svm_search.score(features_test, class_test)))
    print("{:<30} {:<64} {:<24}".format('BaggingClassifier',str(bag_params),bag_search.score(features_test, class_test)))
    print("{:<30} {:<64} {:<24}".format('AdaBoost',str(ada_params),ada_search.score(features_test, class_test)))
    print("{:<30} {:<64} {:<24}".format('Random Forest ',str(rdf_params),rdf_search.score(features_test, class_test)))
    

if __name__ == "__main__":
    main()
