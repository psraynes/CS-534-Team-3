import cv2
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from joblib import load

from load_image import load_raw_image_glcm, load_raw_image_lbp, IMAGE_SIZE

model_path = input("Please provide the path to the saved model: ")
tex_type = input("Was the model created for GLCM or LBP data? ")
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
# Convert from 3d array to 2d array
input_data = []
for x in range(image_data.shape[0]):
    for y in range(image_data.shape[1]):
        input_data.append(image_data[x][y])
        
results = clf_mlp.predict(input_data)

print("Preparing results")
# Converts from 1d array to 3d color array, but only red
results_image = np.empty(shape=IMAGE_SIZE, dtype='int')
for x in range(image_data.shape[0]):
    for y in range(image_data.shape[1]):
        results_image[x][y] = 0 if results[(x * image_data.shape[0]) + y] == 0 else 255
        
results_image_color = np.empty(shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3), dtype=np.uint8)
results_image_color[:,:,0] = 0
results_image_color[:,:,1] = 0
results_image_color[:,:,2] = results_image

print("Displaying Images")
cv2.imshow('source image',image)
cv2.imshow('classified image',results_image_color)

combined = cv2.addWeighted(image, 1, results_image_color, 0.75, 0)
cv2.imshow('overlayed images',combined)
cv2.waitKey(0)
