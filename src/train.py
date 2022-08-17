# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import math

def train_model(model, criterion, optimizer, scheduler, num_epochs=1):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            mlflow.log_metric('loss_'+phase, epoch_loss)
            mlflow.log_metric('acc_'+phase, epoch_acc)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to the training data')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--dataset_size', type=str, default='full', help='Use the full dataset or a small part of it')
    parser.add_argument('--model_output_path', type=str, help='Path to save the model')

    args = parser.parse_args()
    print("===== IMcoming parameters =====")
    print("DATA PATH: ", args.data_path)
    print("NUME EPOCHS: ", args.num_epochs)
    print("Output Path: ", args.model_output_path)

    std_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(), 
            std_normalize])
    dataset = torchvision.datasets.ImageFolder(args.data_path,transform=trans)
    
    # Getting the classnames
    class_names = dataset.classes

    # Splitting the dataset 80/20
    if args.dataset_size == "small":
        dataset = data_utils.Subset(dataset, torch.arange(1000))
    
    dataset_split_size = math.ceil(len(dataset)*0.8)
    trainset, testset = torch.utils.data.random_split(dataset,[dataset_split_size,len(dataset)-dataset_split_size])

    # Load images
    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)
    dataloaders['val'] = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=4)

    # Dataset size
    dataset_sizes = {}
    dataset_sizes['train'] = len(trainset)
    dataset_sizes['val'] = len(testset)

    print("Training set:",dataset_sizes['train'])
    print("Validation set:",dataset_sizes['val'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on: ", device)

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.learning_rate, momentum=args.momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=args.num_epochs)

    # Save the model
    print("saving model")
    torch.save(model_ft, os.path.join(args.model_output_path,'cats-and-docs.pth'))

    # Save the labels
    with open(os.path.join(args.model_output_path,'labels.txt'), 'w') as f:
        f.writelines(["%s\n" % item  for item in class_names])

    print("done")

    

        


