import cv2
import numpy as np
import os
from skimage.feature import graycomatrix, graycoprops


###
# function to take in a path to a mask image, and return an array for the pixels.
# Key for returned array:
# 0 - Not corrosion, black in validation
# 1 - Fair, Red in validation (Maps to 38)
# 2 - Poor, Green in validation (Maps to 75)
# 3 - Severe, Yellow in validation (Maps to 113)

###
def load_mask(path, levels=False):
    picture = cv2.imread(path, 0)
    # If levels of corrosion are needed, map to 1,2,3, otherwise, map nonzero values to 1
    if levels:
        picture[picture == 38] = 1
        picture[picture == 75] = 2
        picture[picture == 113] = 3
    else:
        picture[picture != 0] = 1
    return picture

###
# function to take in a path to a raw image, and return an array for the pixels.
# For each pixel, the array at that location contains the HSV values well as the contrast 
# and correlation for each of 4 directions

###
def load_raw_image(path):
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

###
# function to ask the user for the image and mask directories, then load the files

###
def load_all_files():
    image_folder = input("Please provide the directory containing the images: ")
    mask_folder = input("Please provide the directory containing the masks: ")
    
    # Replace \ with /
    image_folder = image_folder.replace("\\","/")
    mask_folder = mask_folder.replace("\\","/")
    
    # Add / if its missing from directory
    if not (image_folder.endswith("/") or image_folder.endswith("\\")):
        image_folder = image_folder + "/"
    if not (mask_folder.endswith("/") or mask_folder.endswith("\\")):
        mask_folder = mask_folder + "/"
        
    # List all files in both directories, ignore file extension
    image_file_names = os.listdir(image_folder)
    mask_file_names = os.listdir(mask_folder)
    
    missing_masks = set(os.path.splitext(file_name)[0] for file_name in image_file_names).difference(set(os.path.splitext(file_name)[0] for file_name in mask_file_names))
    
    if len(missing_masks) > 0:
        print("The following files do not have masks, omitting them from load:")
        print(missing_masks)
    
    pixel_data = []
    for file_name in image_file_names:
        if os.path.splitext(file_name)[0] not in missing_masks:
            # raw_data = load_raw_image(image_folder + file_name)
            print(image_folder + file_name)
            print(mask_folder + file_name)
            mask = load_mask(mask_folder + file_name)
        
load_all_files()