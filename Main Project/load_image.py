import cv2
import numpy as np
import os
import csv
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from tqdm import tqdm

IMAGE_SIZE = (512,512)

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
    
    # Resize the image to 512x512
    resize_img = cv2.resize(picture, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    
    # If levels of corrosion are needed, map to 1,2,3, otherwise, map nonzero values to 1
    if levels:
        resize_img[resize_img == 38] = 1
        resize_img[resize_img == 75] = 2
        resize_img[resize_img == 113] = 3
    else:
        resize_img[resize_img != 0] = 1
    return resize_img

###
# function to take in a path to a raw image, and return an array for the pixels.
# For each pixel, the array at that location contains the HSV values well as the contrast 
# and correlation for each of 4 directions of GLCM

###
def load_raw_image_glcm(path):
    picture = cv2.imread(path)
    
    # Resize the image to 512x512
    resize_img = cv2.resize(picture, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

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
# function to take in a path to a raw image, and return an array for the pixels.
# For each pixel, the array at that location contains the HSV values well as the contrast 
# and correlation for each of 4 directions of GLCM

###
def load_raw_image_lbp(path):
    picture = cv2.imread(path)
    
    # Resize the image to 512x512
    resize_img = cv2.resize(picture, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    # convert to hsv and grayscale
    hsv_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2HSV)
    grayscale_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)

    # Create LBP data, multiple radius' allows us to better quantify the texture
    lbp1_img = local_binary_pattern(grayscale_img,8,1)
    lbp2_img = local_binary_pattern(grayscale_img,16,2)
    lbp3_img = local_binary_pattern(grayscale_img,24,3)
    
    # Combine HSV and LBP data into one object
    data = np.empty(shape=(hsv_img.shape[0],hsv_img.shape[1],6))

    for x in range(hsv_img.shape[0]):
        for y in range(hsv_img.shape[1]):
            data[x][y] = np.append(hsv_img[x][y], [lbp1_img[x][y], lbp2_img[x][y], lbp3_img[x][y]])
            
    return data

###
# function to ask the user for the image and mask directories, then load the files
# Returns a 2d array containing a unique row for each pixel in all the images scaled to IMAGE_SIZE
# Loads the GLCM data for the image

###
def load_all_files_glcm():
    image_folder = input("Please provide the directory containing the images: ")
    mask_folder = input("Please provide the directory containing the masks: ")
    output_name = input("Please name the output file: ")
    
    # Replace \ with /
    image_folder = image_folder.replace("\\","/")
    mask_folder = mask_folder.replace("\\","/")
    
    # Add / if its missing from directory
    if not (image_folder.endswith("/") or image_folder.endswith("\\")):
        image_folder = image_folder + "/"
    if not (mask_folder.endswith("/") or mask_folder.endswith("\\")):
        mask_folder = mask_folder + "/"
        
    # List all files in both directories
    image_file_names = os.listdir(image_folder)
    mask_file_names = os.listdir(mask_folder)
    
    # Separate the names from the extensions and determine which are missing
    image_file_name_map = {}
    mask_file_name_map = {}
    missing_images = []
    
    for file_name in image_file_names:
        (name, ext) = os.path.splitext(file_name)
        image_file_name_map.update({name: ext})
        
    for file_name in mask_file_names:
        (name, ext) = os.path.splitext(file_name)
        if name not in image_file_name_map.keys():
            missing_images.append(name)
        else:
            mask_file_name_map.update({name: ext})
    
    missing_masks = set(image_file_name_map.keys()).difference(set(mask_file_name_map.keys()))
    
    if len(missing_masks) > 0:
        print("The following files do not have masks, omitting them from load:")
        print(missing_masks)
    
    if len(missing_images) > 0:
        print("The following masks do not have files, omitting them from load:")
        print(missing_images)
        
    # Setup CSV output
    csv_file = open(output_name + ".csv",'w')
    csv_writer = csv.writer(csv_file)
    header = ["uid","h","s","v","con1","cor1","con2","cor2","con3","cor3","con4","cor4","label"]
    csv_writer.writerow(header)
    num_rows = 0
    file_num = 1
    
    progress_bar = tqdm(total=len(image_file_name_map))
    for file_name in image_file_name_map:
        if file_name not in missing_masks:
            raw_data = load_raw_image_glcm(image_folder + file_name + image_file_name_map[file_name])
            mask = load_mask(mask_folder + file_name + mask_file_name_map[file_name])
            
            for x in range(IMAGE_SIZE[0]):
                for y in range(IMAGE_SIZE[1]):
                    uid = file_name + "x" + str(x) + "y" + str(y) # Generate a unique id for this pixel
                    data_row = [uid]
                    data_row.extend(raw_data[x][y].tolist()) # Add the raw data to the list
                    data_row.append(mask[x][y]) # Add the mask data to the list
                    
                    csv_writer.writerow(data_row)
                    csv_file.flush()
                    num_rows = num_rows + 1
                    
                    if num_rows > 5000000:
                        csv_file.close()
                        file_num = file_num + 1
                        csv_file = open(output_name + str(file_num) + ".csv",'w')
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerow(header)
                        num_rows = 0
                        
        progress_bar.update(1)
           
    csv_file.close()         
    return


###
# function to ask the user for the image and mask directories, then load the files
# Returns a 2d array containing a unique row for each pixel in all the images scaled to IMAGE_SIZE
# Loads the LBP data for the image

###
def load_all_files_lbp():
    image_folder = input("Please provide the directory containing the images: ")
    mask_folder = input("Please provide the directory containing the masks: ")
    output_name = input("Please name the output file: ")
    
    # Replace \ with /
    image_folder = image_folder.replace("\\","/")
    mask_folder = mask_folder.replace("\\","/")
    
    # Add / if its missing from directory
    if not (image_folder.endswith("/") or image_folder.endswith("\\")):
        image_folder = image_folder + "/"
    if not (mask_folder.endswith("/") or mask_folder.endswith("\\")):
        mask_folder = mask_folder + "/"
        
    # List all files in both directories
    image_file_names = os.listdir(image_folder)
    mask_file_names = os.listdir(mask_folder)
    
    # Separate the names from the extensions and determine which are missing
    image_file_name_map = {}
    mask_file_name_map = {}
    missing_images = []
    
    for file_name in image_file_names:
        (name, ext) = os.path.splitext(file_name)
        image_file_name_map.update({name: ext})
        
    for file_name in mask_file_names:
        (name, ext) = os.path.splitext(file_name)
        if name not in image_file_name_map.keys():
            missing_images.append(name)
        else:
            mask_file_name_map.update({name: ext})
    
    missing_masks = set(image_file_name_map.keys()).difference(set(mask_file_name_map.keys()))
    
    if len(missing_masks) > 0:
        print("The following files do not have masks, omitting them from load:")
        print(missing_masks)
    
    if len(missing_images) > 0:
        print("The following masks do not have files, omitting them from load:")
        print(missing_images)
        
    # Setup CSV output
    csv_file = open(output_name,'w')
    csv_writer = csv.writer(csv_file)
    header = ["uid","h","s","v","lbp1","lbp2","lbp3","label"]
    csv_writer.writerow(header)
    
    for file_name in image_file_name_map:
        if file_name not in missing_masks:
            raw_data = load_raw_image_lbp(image_folder + file_name + image_file_name_map[file_name])
            mask = load_mask(mask_folder + file_name + mask_file_name_map[file_name])
            
            for x in range(IMAGE_SIZE[0]):
                for y in range(IMAGE_SIZE[1]):
                    uid = file_name + "x" + str(x) + "y" + str(y) # Generate a unique id for this pixel
                    data_row = [uid]
                    data_row.extend(raw_data[x][y].tolist()) # Add the raw data to the list
                    data_row.append(mask[x][y]) # Add the mask data to the list
                    
                    csv_writer.writerow(data_row)
                    csv_file.flush()
           
    csv_file.close()         
    return

if __name__ == "__main__":
    tex_type = input("Would you like to load with GLCM or LBP processing? ")

    if tex_type.casefold() == "glcm":
        load_all_files_glcm()
    else:
        load_all_files_lbp()