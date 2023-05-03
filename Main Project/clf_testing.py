
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

path = "C:/Users/Owner/Desktop/lbp_data/CCSC512Trainlbp.csv"
path_test = "C:/Users/Owner/Desktop/lbp_data/CCSC512Testlbp.csv"
path2 = "C:/Users/willg/Documents/Grad School/Spring23/CS_534/CCSC512Train.csv"
path_test2 = "C:/Users/willg/Documents/Grad School/Spring23/CS_534/CCSC512Test.csv"
df_test = pd.read_csv(path_test2)
df = pd.read_csv(path2)

down_sampled_df = df.sample(n=df.shape[0]//256, random_state=534)

print("Finally loaded the data!")

# features = down_sampled_df[["h","s","v","lbp1","lbp2","lbp3"]]
# labels = down_sampled_df['label']
features = down_sampled_df[["h","s","v","con1","cor1","con2","cor2","con3","cor3","con4","cor4"]]
labels = down_sampled_df['label']

features_test = df_test[["h","s","v","con1","cor1","con2","cor2","con3","cor3","con4","cor4"]]
labels_test = df_test['label']

#clf_knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
clf_mlp = MLPClassifier(hidden_layer_sizes=(100, 40, 30,), max_iter=1000, activation='logistic')

mlp_grid = {'hidden_layer_sizes': [(100, 10, 10,), (100, 30, 10,), (100, 40, 10,),
                                   (100, 10, 30,), (100, 30, 30,), (100, 40, 30,),
                                   (100, 10, 40,), (100, 30, 40,), (100, 40, 40,)],
            'activation': ['identity', 'logistic', 'tanh', 'relu']}

# mlp_search = GridSearchCV(clf_mlp, mlp_grid, scoring='f1', n_jobs=-1, cv=5)
# mlp_search.fit(features, labels)
# mlp_params = mlp_search.best_params_
clf_mlp.fit(features, labels)
print("Training Done")
# mlp_score = mlp_search.score(features_test, labels_test)
# mlp_pred_class = mlp_search.predict(features_test)

mlp_score = clf_mlp.score(features_test, labels_test)
mlp_pred_class = clf_mlp.predict(features_test)

#clf_knn.fit(features, labels)

#print("Nearest Neighbors Training Accuracy:", clf_knn.score(features, labels))

print("Multi-Layer Perceptron Test Accuracy:", mlp_score)

conf_mat = confusion_matrix(labels_test, mlp_pred_class)

cm_display = ConfusionMatrixDisplay(conf_mat).plot()
plt.show()

# print(mlp_params)

