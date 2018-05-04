#from torch.utils.data import Dataset, ImageFolder
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy

data_transform = transforms.Compose([
    transforms.Resize(299).,
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_dir = 'new_data'
image_dataset = datasets.ImageFolder(os.path.join(data_dir, 'heldout'), data_transform)
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=64, shuffle = False, num_workers = 16)
dataset_sizes = len(image_dataset)
class_names = image_datasets.classes

if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_model(model, ):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(inputs)

        if isinstance(outputs, tuple):
            _, preds = torch.max(outputs[0], 1)
            loss = sum((criterion(o,labels) for o in outputs))
        else:
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_corrects += torch.sum(preds == labels.data)

    acc = running_corrects.double() / dataset_sizes

    print('Accuracy: {:.4f}'.format(acc))


model = torch.load('inception_10epochs.pt')

model = model.to(device)



criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=10)

torch.save(model_ft, 'inception_10epochs.pt')
