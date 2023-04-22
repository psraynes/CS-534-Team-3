import torch
import torch.utils.data
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

train_folder = input("Please provide the directory for the training data: ")
test_folder = input("Please provide the directory for the testing data: ")
if not train_folder:
    train_folder = "Group Project 4/complete_mednode_dataset/train/"
if not test_folder:
    test_folder = "Group Project 4/complete_mednode_dataset/test/"

# Transformations to perform on the training and testing data
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomRotation(degrees=(30, 70)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# Load folders into data sets
train_dataset = tv.datasets.ImageFolder(root=train_folder, transform=train_transform)
test_dataset = tv.datasets.ImageFolder(root=test_folder, transform=test_transform)

# Turn data sets into data loaders
batch_sz = 10
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_sz, shuffle=True)

alex = tv.models.AlexNet(num_classes=2,dropout=0) # Note: We need to perform 5 fold cross validation on this dropout value
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(alex.parameters(), lr=0.001, momentum=0.9)

running_loss = 0.0
for i, data in enumerate(train_loader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = alex(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    print(f'[{i + 1:5d}] loss: {running_loss:.3f}')
    running_loss = 0.0
        
            