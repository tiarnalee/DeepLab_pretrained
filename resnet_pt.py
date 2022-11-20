#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 17:16:32 2022

@author: tle19
https://medium.com/nerd-for-tech/image-classification-using-transfer-learning-pytorch-resnet18-32b642148cbe

fully connected layers
https://pythonguides.com/pytorch-fully-connected-layer/
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
import json
import matplotlib as mpl
mpl.use('Agg')
os.chdir('/home/tle19/Desktop/ResNet_pretrained/')
import csv
import json
import sys
import glob
from PIL import Image
from datetime import datetime, date
from torchvision.utils import make_grid
import shutil
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from batchgenerators.utilities.file_and_folder_operations import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from custom_transforms import *


training_validation_path = '/home/tle19/Desktop/ResNet_pretrained/Images/'

classes=['3A', '3B', '3C', '4A', '4B', '4C']
# classes=['3A', '3B', '3C']
# classes=['4A', '4B', '4C']
im_paths, labels=[], []

# Find image paths
_=[im_paths.append(glob.glob(os.path.join(training_validation_path, i, "*"))) for i in classes]
im_paths=np.hstack(im_paths)
# Collect labels from image folders
_=[labels.append(j.split('/')[-2]) for j in im_paths]


# Split into training and test
X_train, X_test, y_train, y_test = train_test_split(im_paths, labels, test_size=0.1, shuffle=True, stratify=labels)
np.save(f'/home/tle19/Desktop/ResNet_pretrained/results/test_cases-{date.today().strftime("%d%B")}{datetime.now().strftime("%H:%M")}.npy', X_test)
# Split into training and val
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/4.5, shuffle=True, stratify=y_train)

print(f'There are {len(X_train)} training images and {len(X_val)} test images ({len(X_train)+len(X_val)} total)')

#%%
"""Set dataloader"""

class hair_dataset(Dataset):
    def __init__(self, root_dir, transforms):
      self.root_dir = root_dir
      self.transforms = transforms

    def __len__(self):
        return len(self.root_dir)

    def __getitem__(self, index):
      # Select sample
      image = Image.open(self.root_dir[index]).convert("RGB")
      label=self.convert_labels(self.root_dir[index])
      sample={'image': image, 'label': label} #Make image and label a dict pair

      transformed_im = self.transforms(sample) #Perform transforms

      return (transformed_im['image'], transformed_im['label'])

# Convert alphabetical labels to numbers (starting at 1)
    def convert_labels(self, dir):
      classes=['3A', '3B', '3C', '4A', '4B', '4C']

      label=np.where(dir.split('/')[-2]==np.array(classes))[0][0] #Labels between 1-6, set label as index in list
      return label

# Calculate normalisation parameters
batch_size = 8
img_height, img_width = 400, 400

#%%
"""Create datasets"""
means= [0.449891447249903, 0.34121201611416146, 0.2946232938238485]
stds= [0.31363333313315633, 0.25191345914521435, 0.22231266962369908]

train_transforms = transforms.Compose([
    RandomFlip(),
    RandomRotate(),
    RandomCrop(sizes=(400)), #220
    RandomGaussianBlur(),
    # RandomInv(),
    Random_Brightness(),
    Random_Contrast(),
    Random_Saturation(),
    Adjust_Gamma(),
    Adjust_Sharpness(),
    ToTensor(),
    Normalise(means=means,stds=stds,),
    Rescale_pixel_values(),
    Resize(sizes=img_height),
])  

val_transforms = transforms.Compose([
  ToTensor(),
  Normalise(means=means,stds=stds,),
  Rescale_pixel_values(),
  Resize(sizes=img_height),
])  
    
#define class sampler
# This can be removed
def find_sampler(training_samples, classes, batch_size):
    _,counts=np.unique(training_samples, return_counts=True) #find no of unique labels
    class_weights=[sum(counts)/c for c in counts] #find weights
    # class_weights[1]*=0.5
    labels=[np.where(training_samples[i]==np.array(classes))[0][0] for i in range(len(training_samples))]#convert hair labels to ints
    example_weights=[class_weights[i] for i in labels]#assign weight to each sample
    sampler=WeightedRandomSampler(example_weights, len(training_samples), replacement=False)
    return sampler
    
#load training data
training_sampler=find_sampler(y_train, classes, batch_size)
train_dataset = hair_dataset(X_train, transforms=train_transforms)
train_dl = DataLoader(train_dataset, sampler=training_sampler, batch_size=batch_size, num_workers=4, pin_memory=True)

#load val data
val_sampler=find_sampler(y_val, classes, batch_size)
val_dataset = hair_dataset(X_val, transforms=val_transforms)
val_dl = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size, num_workers=4, pin_memory=True)

#PLot a batch 
# data_iter = iter(train_dl)
# images, labels = data_iter.next()

# grid = make_grid(images[0], padding=2)
# fig = plt.figure(figsize=(20, 20))
# plt.imshow(grid.numpy().transpose((1, 2, 0)))
# plt.axis('off')
#%% TRAINING
#iterations needed for each epoch
iterations=int(np.ceil(len(X_train)/batch_size))

epochs=150
print(f'{epochs} epochs and {iterations} iterations')

# OPTIMISER PARAMETERS
lr = 0.001 # authors cite 0.1
momentum = 0.9
weight_decay = 1e-4 #0.0001 

# LEARNING RATE ADJUSTMENT
# milestones = [round(0.5*epochs/iterations), round(0.75*epochs/iterations)]
milestones = [25, 50]
milestones_on=False
print(f'LR reduced by factor of 10 at epochs {milestones}' if milestones_on else 'No LR reduction')
# Divide learning rate by 10 at each milestone
gamma = 0.1
print('Using ', classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object
model = models.googlenet(pretrained=True)   #load resnet18 model

model.fc = nn.Linear(model.fc.in_features, len(np.unique(classes)), bias=True) #replace 2 output notes with no of classes
model = model.to(device) 
# print(model)
#%%
criterion = nn.CrossEntropyLoss()  #(set loss function)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

#save parameters
params={ 'lr': lr, 'momentum': momentum, 
        'weight_decay': weight_decay, 'milestones on/off': milestones_on, 'milestones': milestones, 'epochs': epochs, 'gamma': gamma, 'labels_used': classes}
save_json(params, '/home/tle19/Desktop/ResNet_pretrained/results/params.json')

#set save path
save_path = f'/home/tle19/Desktop/ResNet_pretrained/results/resnet50-{date.today().strftime("%d%B")}{datetime.now().strftime("%H:%M")}.pth'

#copy current transforms file
_=shutil.copy('/home/tle19/Desktop/ResNet_pretrained/custom_transforms.py', '/home/tle19/Desktop/ResNet_pretrained/results/custom_transforms.py')

best_acc=0.
best_acc1=0.
cols  = ['epoch', 'train_err', 'val_err', 'train_loss', 'val_loss']
results_df = pd.DataFrame(columns=cols).set_index('epoch')
for epoch in range(epochs): #(loop for every epoch)
    t0 = time.time()
    print("Epoch {}".format(epoch)) #(printing message)
    """ Training Phase """
    model.train()    #(training model)
    running_loss, running_corrects = 0. ,0  #(set loss 0)
    # load a batch data of images
    for i, (inputs, labels) in enumerate(train_dl):
        inputs = inputs[0].to(device)
        labels = labels.to(device) 
        # forward inputs and get output
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        # get loss value and update the network weights
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        if milestones_on==True:
            scheduler.step()
    train_loss = running_loss / len(train_dataset)
    train_acc = running_corrects / len(train_dataset) * 100.
    print('[Train #{}] Loss: {:.4f} Acc: {:.4f}%'.format(epoch, train_loss, train_acc))
    
    """ Testing Phase """
    model.eval()
    with torch.no_grad():
        running_loss, running_corrects = 0. ,0  #(set loss 0)

        for inputs, labels in val_dl:
            inputs = inputs[0].to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        val_loss = running_loss / len(val_dataset)
        val_acc = running_corrects / len(val_dataset) * 100.
        print('[Test #{}] Loss: {:.4f} Acc: {:.4f}%'.format(epoch, val_loss, val_acc))
    
    # if val_acc>80:
        # optimizer = optim.SGD(model.parameters(), lr=lr/10, momentum=momentum, weight_decay=weight_decay)
    
    results_df.loc[epoch] = [train_acc.cpu(), val_acc.cpu(), train_loss, val_loss] 
    print('This epoch took {} seconds'.format(np.round(time.time() - t0, 3)))
    if (val_acc > best_acc):
        torch.save(model.state_dict(), save_path)
        best_acc = val_acc
    if train_acc>best_acc1:
        best_acc1=train_acc
        
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize = (20,10), tight_layout = True)
    # _=plt.subplot(121)
    plt.rcParams['font.size'] = '20'
    ax1.plot(np.arange(len(results_df.train_err.values)), results_df.train_err.values, label='train')
    ax1.plot(np.arange(len(results_df.train_err.values)), results_df.val_err.values, label='val')
    ax1.legend(loc='upper left')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('Acc (%)')
    txt='best train score: ' + str(np.round(best_acc1.cpu().detach().numpy(),1)) + '\nbest val score: ' + str(np.round(best_acc.cpu().detach().numpy(),1))
    ax1.text(0.5, 1, txt, horizontalalignment='center', verticalalignment='center',  transform=ax1.transAxes)

    #plot loss    
    # _=plt.subplot(122)
    ax2.plot(np.arange(len(results_df.train_loss.values)), results_df.train_loss.values, label='train', color='r')
    ax2.plot(np.arange(len(results_df.train_loss.values)), results_df.val_loss.values, label='val', color='g')
    ax2.legend(loc='upper right')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('CE Loss')
    txt='best train loss: ' + str(np.round(min(results_df.train_loss.values),3)) + '\nbest val loss: ' + str(np.round(min(results_df.val_loss.values),3))
    ax2.text(1.6, 1, txt, horizontalalignment='center', verticalalignment='center',  transform=ax1.transAxes)
    plt.savefig('/home/tle19/Desktop/ResNet_pretrained/results/progress.png')
    plt.close(fig)
        
_
