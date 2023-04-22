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
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
test_transform = transforms.Compose([
    transforms.Resize((512, 512)),
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
batch_sz = 1
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_sz, shuffle=True)

alex = tv.models.AlexNet(num_classes=2, dropout=0) # Note: We need to perform 5 fold cross validation on this dropout value
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(alex.parameters(), lr=0.01, momentum=0)

for epoch in range(5):  # loop over the dataset multiple times
    running_loss = 0.0
    alex.train(True)
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
        if i % 5 == 4:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 5:.3f}')
            running_loss = 0.0
            
    alex.train(False)
        
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = alex(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')
            