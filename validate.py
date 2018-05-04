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
    transforms.Resize(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_dir = 'new_data'
image_dataset = datasets.ImageFolder(os.path.join(data_dir, 'heldout'), data_transform)
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=32, shuffle = True, num_workers = 16)
dataset_sizes = len(image_dataset)
class_names = image_dataset.classes

if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

def test_model(model):
    model.eval()

    running_corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        if isinstance(outputs, tuple):
            _, preds = torch.max(outputs[0], 1)
        else:
            _, preds = torch.max(outputs, 1)

        print(preds)
        print(labels.data)
        running_corrects += torch.sum(preds == labels.data)
        print(running_corrects.double())

    acc = running_corrects.double() / dataset_sizes

    print('Accuracy: {:.4f}'.format(acc))


model = torch.load('inception_25epochs.pt')
model = model.to(device)

test_model(model)
