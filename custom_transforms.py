import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as T

class Rescale_pixel_values(object):
    def __call__(self, sample):
        img = sample['image']
        img=torch.div((img-torch.min(img)),(torch.max(img)-torch.min(img)))
        return {'image': img,'label': sample['label']}

class Normalise(object):
    def __init__(self, means, stds):
        self.means = means
        self.stds=stds
        
    def __call__(self, sample):
        img = sample['image']
        img=T.Normalize(self.means,self.stds)(img)
        return {'image': img,'label': sample['label']}
        
class Resize(object):
    def __init__(self, sizes=300):
        self.sizes = sizes

    def __call__(self, sample):
        img = sample['image']
        img= T.functional.resize(img, size=(self.sizes, self.sizes)),
        return {'image': img,'label': sample['label']}
        
class RandomFlip(object):
    def __call__(self, sample):
        img = sample['image']
        if np.random.randn() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.randn() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)   
        return {'image': img,'label': sample['label']}
        
class RandomRotate(object):
    def __call__(self, sample):
        img = sample['image']
        if np.random.randn() < 0.3:
            rotate_degree = random.uniform(-180, 180)
            img = img.rotate(rotate_degree, Image.BILINEAR)
        return {'image': img,'label': sample['label']}
        
class RandomCrop(object):
    def __init__(self,sizes):
        self.sizes = sizes
        
    def __call__(self, sample):
        img = sample['image']
        if (np.random.randn() < 0.3):
          img = T.RandomCrop(size=(self.sizes), pad_if_needed=True)(img)
        return {'image': img,'label': sample['label']}
        
class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        if np.random.randn() < 0.15:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        return {'image': img,'label': sample['label']}
        
class RandomInv(object):
  def __call__(self, sample):
    img = sample['image']
    if np.random.randn() < 0.1:
      img = T.RandomInvert()(img)
    return {'image': img,'label': sample['label']}
        
class Random_Brightness(object):
  def __call__(self, sample):
    img = sample['image']
    if np.random.randn() < 0.2:
      img = T.ColorJitter(brightness=(0.7, 1.3), hue=0, saturation=0, contrast=0)(img)
    return {'image': img,'label': sample['label']}

class Random_Contrast(object):
  def __call__(self, sample):
    img = sample['image']
    if np.random.randn() < 0.15: #0.15
      img = T.ColorJitter(brightness=0, hue=0, saturation=0, contrast=(0.65, 1.5))(img)
    return {'image': img,'label': sample['label']}

class Random_Saturation(object):
  def __call__(self, sample):
    img = sample['image']
    if np.random.randn() < 0.15: #.15
      img = T.ColorJitter(brightness=0, hue=0, saturation=(0.75, 1.25), contrast=0)(img)
    return {'image': img,'label': sample['label']}

class Adjust_Gamma(object):
  def __call__(self, sample):
    img = sample['image']
    if np.random.randn() < 0.2: #0.15
      img = T.functional.adjust_gamma(img, gain= random.uniform(0, 1), gamma=random.uniform(0.7, 1.5))
    return {'image': img,'label': sample['label']}
    
class Adjust_Sharpness(object):
  def __call__(self, sample):
    img = sample['image']
    if np.random.randn() < 0.3: #0.15
      img = T.RandomAdjustSharpness(sharpness_factor=random.uniform(0, 3), p=1)(img)
    return {'image': img,'label': sample['label']}
       
class ToTensor(object):
    def __call__(self, sample):
        img = sample['image']
        img = T.ToTensor()(img)

        return {'image': img,
                'label': sample['label']}