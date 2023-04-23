import torch
import torch.utils.data
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from cross_validation import reset_weights


def main():
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

    k_folds = 5
    num_epochs = 1
    batch_sz = 1
    criterion = nn.CrossEntropyLoss()

    # Results from fold for variable dropouts
    results = {}

    # Load folders into data sets
    train_dataset = tv.datasets.ImageFolder(root=train_folder, transform=train_transform)
    test_dataset = tv.datasets.ImageFolder(root=test_folder, transform=test_transform)
    dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Note: We need to perform 5 fold cross validation on this dropout value
    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print("Dropout: ", fold*0.25)

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Turn data sets into data loaders
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_sz,
            sampler=train_subsampler)
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_sz,
            sampler=test_subsampler)

        alex = tv.models.AlexNet(num_classes=2, dropout=(fold*0.25))
        optimizer = optim.SGD(alex.parameters(), lr=0.01, momentum=0)
        alex.apply(reset_weights)

        for epoch in range(num_epochs):  # loop over the dataset multiple times
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

            print("End of epoch :------------")
            # Saving the model
        save_path = f'./model-fold-{fold}.pth'
        torch.save(alex.state_dict(), save_path)

        # Evaluation for this fold
        correct, total = 0, 0
        with torch.no_grad():

            # Iterate over the test data and generate predictions
            for i, data in enumerate(train_loader, 0):
                # Get inputs
                inputs, targets = data

                # Generate outputs
                outputs = alex(inputs)

                # Set total and correct
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            # Print accuracy
            print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
            print('--------------------------------')
            results[fold] = 100.0 * (correct / total)
        print("End of Fold :------------")

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR BEST FOLD')
    print('--------------------------------')
    best = [i for i in results if results[i] == max(results.values())]
    best_fold = int(best[0])
    best_val = results[best[0]]
    best_dropout = best_fold*0.25
    fold_sum = 0.0
    drop = 0
    print(f'Fold {best_fold}, Dropout: {best_dropout}): {best_val} %')

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
    
if __name__ == "__main__":
    main()
            