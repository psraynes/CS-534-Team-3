import os
import shutil
from random import sample

# This file will take your dataset and split it into a testing and training dataset.

def split_test_train():
    image_folder = input("Please provide the directory containing the images: ")
    mask_folder = input("Please provide the directory containing the masks: ")
    output_folder = input("Where would you like the testing and training folders to be created? ")
    percet_test = input("What percentage of the data should be used for testing? ")
    
    # Replace \ with /
    image_folder = image_folder.replace("\\","/")
    mask_folder = mask_folder.replace("\\","/")
    output_folder = output_folder.replace("\\","/")
    
    # Add / if its missing from directory
    if not (image_folder.endswith("/") or image_folder.endswith("\\")):
        image_folder = image_folder + "/"
    if not (mask_folder.endswith("/") or mask_folder.endswith("\\")):
        mask_folder = mask_folder + "/"
    if not (output_folder.endswith("/") or output_folder.endswith("\\")):
        output_folder = output_folder + "/"
        
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
        
    train_images_folder = output_folder + "Train/Images/"
    train_masks_folder = output_folder + "Train/Masks/"
    test_images_folder = output_folder + "Test/Images/"
    test_masks_folder = output_folder + "Test/Masks/"
    
    if not os.path.exists(train_images_folder):
        os.makedirs(train_images_folder)
    if not os.path.exists(train_masks_folder):
        os.makedirs(train_masks_folder)
    if not os.path.exists(test_images_folder):
        os.makedirs(test_images_folder)
    if not os.path.exists(test_masks_folder):
        os.makedirs(test_masks_folder)
        
    images_not_missing_masks = [i for i in list(image_file_name_map.keys()) if i not in missing_masks]
    test_dataset = sample(images_not_missing_masks, int(len(images_not_missing_masks)*float(percet_test)*0.01))
    
    for file_name in image_file_name_map:
        if file_name not in missing_masks:
            if file_name not in test_dataset:
                shutil.copy(image_folder + file_name + image_file_name_map[file_name], train_images_folder + file_name + image_file_name_map[file_name])
                shutil.copy(mask_folder + file_name + mask_file_name_map[file_name], train_masks_folder + file_name + mask_file_name_map[file_name])
            else:
                shutil.copy(image_folder + file_name + image_file_name_map[file_name], test_images_folder + file_name + image_file_name_map[file_name])
                shutil.copy(mask_folder + file_name + mask_file_name_map[file_name], test_masks_folder + file_name + mask_file_name_map[file_name])
        

if __name__ == "__main__":
    split_test_train()