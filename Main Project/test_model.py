import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from joblib import load
import matplotlib.pyplot as plt

from load_image import load_raw_image_glcm, load_raw_image_lbp, IMAGE_SIZE

model_path = input("Please provide the path to the saved model: ")
tex_type = input("Was the model created for GLCM or LBP data? ")
test_data = input("Please provide the path to a dataset file: ")

print("Loading model")
clf_mlp = load(model_path)

print("Loading Dataset")
df_test = pd.read_csv(test_data)

features = df_test[["h","s","v","con1","cor1","con2","cor2","con3","cor3","con4","cor4"]]
labels = df_test['label']

print("Testing Dataset")

mlp_score = clf_mlp.score(features, features)
mlp_pred_class = clf_mlp.predict(features)

print("Multi-Layer Perceptron Test Accuracy:", mlp_score)

conf_mat = confusion_matrix(features, mlp_pred_class)

cm_display = ConfusionMatrixDisplay(conf_mat).plot()
plt.show()