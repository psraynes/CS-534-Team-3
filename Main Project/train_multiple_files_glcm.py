import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier

paths = []
done = False

while not done:
    text = input('Please provide a file or type "done" to start training: ')
    if text == "done":
        done = True
    else:
        paths.append(text)
        
clf_mlp = MLPClassifier(hidden_layer_sizes=(100, 50, 25, ))
        
for path in paths:
    df = pd.read_csv(path)
    print("Loaded data from " + path)
    
    features = df[["h","s","v","con1","cor1","con2","cor2","con3","cor3","con4","cor4"]]
    labels = df['label']
    
    clf_mlp.partial_fit(features, labels)
    print("Finished fitting data from " + path)
    
print("Finished fitting all data")
