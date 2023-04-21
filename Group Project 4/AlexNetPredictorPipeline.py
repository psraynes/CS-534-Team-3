import os
from torch.utils import data
from torchvision import datasets
import torchvision.transforms as transforms

melanoma_folder = input("Please provide the directory for the Melanoma pictures: ")
naevus_folder = input("Please provide the directory for the Naevus pictures: ")
if not melanoma_folder:
    melanoma_folder = "Group Project 4/complete_mednode_dataset/melanoma/"
if not naevus_folder:
    naevus_folder = "Group Project 4/complete_mednode_dataset/naevus/"

melanoma_dataset = datasets.ImageFolder(melanoma_folder)
naevus_dataset = datasets.ImageFolder(naevus_folder)

melanoma_trainset = data.Subset(melanoma_dataset, range(50))
melanoma_testset = data.Subset(melanoma_dataset, range(50,70))

naevus_trainset = data.Subset(naevus_dataset, range(50))
naevus_testset = data.Subset(naevus_dataset, range(50,70))

train_set = data.ConcatDataset([melanoma_trainset, naevus_trainset])
test_set = data.ConcatDataset([melanoma_testset, naevus_testset])

