import cv2
from skimage.feature import graycomatrix, graycoprops
import numpy as np

def loadRawImage(path):
    picture = cv2.imread(path)
    
    # Resize the image to 512x512
    resize_img = cv2.resize(picture, (512,512), interpolation=cv2.INTER_AREA)

    # convert to hsv and grayscale
    hsv_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2HSV)
    grayscale_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)

    # Break image into 16x16 patches -> 32 patches per side -> 1024 patches total
    imgheight = grayscale_img.shape[0]
    imgwidth = grayscale_img.shape[1]

    patch_per_side = 32
    patch_height = imgheight//patch_per_side
    patch_widtth = imgwidth//patch_per_side

    tiles = np.empty(shape=(imgheight//patch_height, imgwidth//patch_widtth, patch_height, patch_widtth), dtype='int')

    for x in range(imgheight):
        for y in range(imgwidth):
            tiles[x//patch_height][y//patch_widtth][x % patch_height][y % patch_widtth] = grayscale_img[x][y]
            
    # Generate a GLCM for each patch
    num_glcm_props = 2 # This represents the number of different glcm properties we read, used to make expanding that number easier
    glcm_data = np.empty(shape=(patch_per_side, patch_per_side, num_glcm_props*4), dtype=float)

    for x in range(patch_per_side):
        for y in range(patch_per_side):
            # We create a glcm for a distance of 1 in 4 direction representing N, NE, E, SE
            glcm = graycomatrix(tiles[x][y], [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)
            
            contrast = graycoprops(glcm, 'contrast')
            correlation = graycoprops(glcm, 'correlation')

            for i in range(0,num_glcm_props*4,num_glcm_props):
                glcm_data[x][y][i] = contrast[0][i//num_glcm_props]
                glcm_data[x][y][i+1] = correlation[0][i//num_glcm_props]
                
    # Combine HSV and GLCM data into one object
    data = np.empty(shape=(hsv_img.shape[0],hsv_img.shape[1],(num_glcm_props*4) + 3))

    for x in range(hsv_img.shape[0]):
        for y in range(hsv_img.shape[1]):
            data[x][y] = np.concatenate((hsv_img[x][y], glcm_data[x//patch_height][y//patch_widtth]))
            
    return data
        
path = "C:/Users/psray/Documents/AI Rust Pictures/Corrosion Condition State Classification/original/Train/images/0.jpeg"
loadRawImage(path)

