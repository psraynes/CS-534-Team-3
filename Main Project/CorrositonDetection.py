import cv2
from skimage.feature import graycomatrix, graycoprops
import numpy as np

# todo: code for loading all images

path = "C:/Users/Owner/Desktop/16624663/Corrosion Condition State Classification/Corrosion Condition State Classification/512x512/Train/images_512/0.jpeg"
picture = cv2.imread(path)


# convert to hsv
hsv_img = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)

grayscale_img = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)

# Break image into 16x16 patches -> 32 patches per side -> 1024 patches total
imgheight = grayscale_img.shape[0]
imgwidth = grayscale_img.shape[1]

patch_per_side = 32

M = imgheight//patch_per_side
N = imgwidth//patch_per_side

img = grayscale_img

img_ht = imgheight
img_wd = imgwidth
tile_ht = M
tile_wd = N

tiles = np.empty(shape=(img_ht//tile_ht, img_wd//tile_wd, tile_ht, tile_wd), dtype='int')

for x in range(img_ht):
    for y in range(img_wd):
        tiles[x//tile_ht][y//tile_wd][x % tile_ht][y % tile_wd] = img[x][y]

print(tiles[0][0])
print(grayscale_img)
# cv2.imshow('img', grayscale_img)
# cv2.waitKey(0)

glcm_data = np.empty(shape=(patch_per_side, patch_per_side, 8), dtype=float)

for x in range(patch_per_side):
    for y in range(patch_per_side):
        glcm = graycomatrix(tiles[x][y], [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)
        
        contrast = graycoprops(glcm, 'contrast')
        correlation = graycoprops(glcm, 'correlation')

        for i in range(0,8,2):
            glcm_data[x][y][i] = contrast[0][i//2]
            glcm_data[x][y][i+1] = correlation[0][i//2]

data = np.empty(shape=(hsv_img.shape[0],hsv_img.shape[1],11))

for x in range(hsv_img.shape[0]):
    for y in range(hsv_img.shape[1]):
        data[x][y] = np.concatenate((hsv_img[x][y], glcm_data[x//M][y//N]))
        
print(data)
