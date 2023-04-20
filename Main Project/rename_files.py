import os

folder = input("What folder would you like to rename? ")
folder = folder.replace("\\","/")
if not folder.endswith("/"):
    folder = folder + "/"
    
for file_name in os.listdir(folder):
    # Separate name from label suffix from extension
    (name, ext) = os.path.splitext(file_name)
    split_name = name.split('_')
    
    # Rename to remove label suffix
    os.rename(folder + file_name, folder + split_name[0] + ext)
    
