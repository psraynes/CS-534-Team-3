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
df_test = pd.read_csv(path_test)
df = pd.read_csv(path)

path2 = "C:/Users/Owner/Desktop/glcm_data/CCSC512Train.csv"
path_test2 = "C:/Users/Owner/Desktop/glcm_data/CCSC512Test.csv"
df_test2 = pd.read_csv(path_test2)
df2 = pd.read_csv(path2)

df[
    ["con1", "cor1", "con2", "cor2", "con3", "cor3", "con4", "cor4"]
] = df2[
    ["con1", "cor1", "con2", "cor2", "con3", "cor3", "con4", "cor4"]
]
df_test[
    ["con1", "cor1", "con2", "cor2", "con3", "cor3", "con4", "cor4"]
] = df_test2[
    ["con1", "cor1", "con2", "cor2", "con3", "cor3", "con4", "cor4"]
]

down_sampled_df = df.sample(n=df.shape[0] // 512, random_state=534)

features = down_sampled_df[
    ["h", "s", "v", "lbp1", "lbp2", "lbp3"#, "con1", "cor1", "con2", "cor2", "con3", "cor3", "con4", "cor4"
     ]]
labels = down_sampled_df['label']

features_test = df_test[
    ["h", "s", "v", "lbp1", "lbp2", "lbp3"#, "con1", "cor1", "con2", "cor2", "con3", "cor3", "con4", "cor4"
     ]]
labels_test = df_test['label']
print("Finally loaded the data!")

# Reported best: {'activation': 'identity', 'hidden_layer_sizes': (100, 40, 10)}, Test Accuracy: 0.1767406538909036
clf_mlp = MLPClassifier(hidden_layer_sizes=(100, 40, 30,), max_iter=1000, activation='logistic')

mlp_grid = {'hidden_layer_sizes': [(10,), (50,), (100,), (100, 10,), (100, 10, 10), (100, 20, 10),
                                   (100, 20, 20), (100, 10, 10, 10,), (50, 10, 10, 5,), (100, 40, 30, )],
            'activation': ['identity', 'logistic', 'tanh', 'relu']}

mlp_search = GridSearchCV(clf_mlp, mlp_grid, scoring='f1', n_jobs=-1, cv=5)
print("Starting CV")
mlp_search.fit(features, labels)
print('Done with CV')
mlp_params = mlp_search.best_params_
mlp_score = mlp_search.score(features_test, labels_test)
mlp_pred_class = mlp_search.predict(features_test)

print("Multi-Layer Perceptron Test Accuracy:", mlp_score)
print(mlp_params)

conf_mat = confusion_matrix(labels_test, mlp_pred_class)

cm_display = ConfusionMatrixDisplay(conf_mat).plot()
plt.show()
