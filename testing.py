#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 23:13:40 2022

@author: tle19
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import json
import shutil
import pandas as pd
from tabulate import tabulate
os.chdir('/home/tle19/Desktop/ResNet_pretrained/')
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from batchgenerators.utilities.file_and_folder_operations import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from custom_transforms import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object
w=[]
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
      if label==1 or label==4:
            self.transforms= transforms.Compose([
              RandomFlip(),
              RandomRotate(),
              ToTensor(),
              Normalise(means=means,stds=stds,),
              Rescale_pixel_values(),
              Resize(sizes=img_height),
            ]) 
      transformed_im = self.transforms(sample) #Perform transforms
      return (transformed_im['image'], transformed_im['label'])

# Convert alphabetical labels to numbers (starting at 1)
    def convert_labels(self, dir):
      classes=['3A', '3B', '3C', '4A', '4B', '4C']

      label=np.where(dir.split('/')[-2]==np.array(classes))[0][0] #Labels between 1-6, set label as index in list
      return label

# Calculate normalisation parameters
img_height, img_width = 400, 400

means= [0.449891447249903, 0.34121201611416146, 0.2946232938238485]
stds= [0.31363333313315633, 0.25191345914521435, 0.22231266962369908]

val_transforms = transforms.Compose([
  ToTensor(),
  Normalise(means=means,stds=stds,),
  Rescale_pixel_values(),
  Resize(sizes=img_height),
])  

classes=['3A', '3B', '3C', '4A', '4B', '4C']
results=[]

# test_cases=np.load(max(glob.glob('/home/tle19/Desktop/ResNet_pretrained/results/*npy'), key=os.path.getctime))
test_dataset = hair_dataset(test_cases, transforms=val_transforms)
test_dl = DataLoader(test_dataset, batch_size=len(test_cases), shuffle=False, num_workers=4, pin_memory=True)

data_iter = iter(test_dl)
images, labels = data_iter.next()

model = models.googlenet(pretrained=True)   #load resnet18 model
model.fc = nn.Linear(model.fc.in_features, len(np.unique(labels)), bias=True) #(num_of_class == 2) model.fc.in_features
   
model.load_state_dict(torch.load(max(glob.glob('/home/tle19/Desktop/ResNet_pretrained/results/*.pth'), key=os.path.getctime)))

model.to(device)
    
with open('/home/tle19/Desktop/ResNet_pretrained/results/params.json') as f:

    d = json.load(f)

    ##Testing
    model.eval()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=d['lr'], momentum=d['momentum'])
    with torch.no_grad():
        running_loss = 0.
        running_corrects = 0
        for i, (inputs, labels) in enumerate(test_dl):
            inputs = inputs[0].to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_corrects += torch.sum(preds == labels.data)
        epoch_acc = running_corrects / len(test_dataset) * 100.
        # print('[Test #{}] Acc: {:.4f}%'.format(i, epoch_acc))
        # print('[Test #{}] Acc: {:.4f}%'.format(x, epoch_acc))
        
    
    YT, YP=[], []
    for i in range(len(preds)):
        YP.append(classes[preds.cpu().numpy()[i]])
        YT.append(classes[labels.cpu().numpy()[i]])
    
    # labels_used=classes
    labels_used=d['labels_used']
    
    if labels_used==['4A', '4B', '4C']:
        YT=['4'+ YT[i][-1:] for i in range(len(YT))]
        YP=['4'+ YP[i][-1:] for i in range(len(YP))]
        
    cm=confusion_matrix(np.hstack(YT), np.hstack(YP), labels=labels_used, normalize='true')
  
    _ = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_used).plot()

    plt.title(str(x) + '/18 frozen layers', fontsize=20)

    # plt.savefig('/home/tle19/Desktop/ResNet_pretrained/results/test_CM.png')
    results.append(np.hstack((x, np.round(cm.diagonal()/cm.sum(axis=1) ,3), epoch_acc.cpu().detach().numpy())))
    YT=np.array(YT)
    YP=np.array(YP)
    wrong=np.array([i.split('/')[-2:] for i in test_cases[np.where(YP!=YT)]])
    
 
    #%%
im_paths=[]

# Find image paths
_=[im_paths.append(glob.glob(os.path.join('/home/tle19/Desktop/ResNet_pretrained/Model hair pics/', i, "*"))) for i in labels_used]
im_paths=np.hstack(im_paths)

# model_test_cases=np.load('/home/tle19/Desktop/ResNet_pretrained/model_pics.npy')
model_dataset = hair_dataset(im_paths, transforms=val_transforms)
model_dl = DataLoader(model_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
# with open('/home/tle19/Desktop/ResNet_pretrained/results/params.json') as f:
    # d = json.load(f)

##Testing
model.eval()
with torch.no_grad():
    running_loss = 0.
    running_corrects = 0
    for i, (inputs, labels) in enumerate(model_dl):
        inputs = inputs[0].to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(test_dataset)
    epoch_acc = running_corrects / len(test_dataset) * 100.
    print('[Test #{}] Loss: {:.4f} Acc: {:.4f}%'.format(i, epoch_loss, epoch_acc))

YT, YP=[], []
for i in range(len(preds)):
    YP.append(classes[preds.cpu().numpy()[i]])
    YT.append(classes[labels.cpu().numpy()[i]])

cm=confusion_matrix(np.hstack(YT), np.hstack(YP), labels=labels_used, normalize='true')
_ = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_used).plot()
plt.title('Model hair photos', fontsize=20)
print(tabulate(np.vstack((labels_used, np.round(cm.diagonal()/cm.sum(axis=1) ,3)))))
plt.savefig('/home/tle19/Desktop/ResNet_pretrained/results/model_test_CM.png')

          
