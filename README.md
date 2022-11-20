# DeepLab_pretrained

## Training the network
To train the network, run:
  ```python resnet_pt.py```

The dataloading and model training is performed in this script. The parameters, model, progress graph and a copy of the current transforms will be saved in the `results` folder

## Relevant files
###
- custom_transforms.py: contains data augmentation classes
- testing.py: find classification accuracy on test sets of hair pics from the internet and from models. NB: there are very few photos of hair from real models so results are likely to be skewed by small sample size
- model_labels: labels for model hair pics. B, T, S correspond to back, top and sides
- Model hair pics: contains images of models' hair. NB: there was no 3A model image so a sample image is used instead. Ignore results for 3A
