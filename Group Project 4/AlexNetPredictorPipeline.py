import os
from sklearn.model_selection import train_test_split

melanoma_folder = input("Please provide the directory for the Melanoma pictures: ")
naevus_folder = input("Please provide the directory for the Naevus pictures: ")
if not melanoma_folder:
    melanoma_folder = "Group Project 4/complete_mednode_dataset/melanoma/"
if not naevus_folder:
    naevus_folder = "Group Project 4/complete_mednode_dataset/naevus/"

melanoma_files = os.listdir(melanoma_folder)
naevus_files = os.listdir(naevus_folder)

melanoma_train, melanoma_test, naevus_train, naevus_test = train_test_split(melanoma_files, naevus_files, train_size=50)

