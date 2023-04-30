import cv2
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from joblib import load

from load_image import load_raw_image_glcm, load_raw_image_lbp, IMAGE_SIZE

model_path = input("Please provide the path to the saved model: ")
tex_type = input("Was the model created for GLCM or LBP data?")
image_path = input("Please provide the path to the image you would like to test: ")

print("Loading model")
clf_mlp = load(model_path)

print("Loading image")
image = picture = cv2.imread(image_path)
if tex_type.casefold() == "glcm":
    image_data = load_raw_image_glcm(image_path)
else:
    image_data = load_raw_image_lbp(image_path)

print("Classifying image")
results = clf_mlp.predict(image_data)

print("Preparing results")
# Converts from 1d array to 
results_image = np.empty(shape=IMAGE_SIZE, dtype='int')
for x in range(IMAGE_SIZE[0]):
    for y in range(IMAGE_SIZE[1]):
        results_image[x][y] = results[(x * IMAGE_SIZE[1]) + y]*255
results_image_color = cv2.cvtColor(results_image,cv2.COLOR_GRAY2RGB)
results_image = results_image_color[:,:,2]

print("Displaying Images")
cv2.imshow('source image',image)  
cv2.imshow('grayscale image',results_image)

combined = cv2.addWeighted(image, 0, results_image, 1)
