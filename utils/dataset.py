import random
from torch.utils.data import DataLoader, Dataset
import cv2
import torch
from os.path import join
import os
import numpy as np


class TrainImageDataset(Dataset):

  def __init__(self, csv, img_folder, transform=None, min_class_num=0, img_size=(256, 256)):
    self.transform = transform
    self.img_folder = img_folder
    self.img_size = img_size
    
    class_counts = csv['Y'].value_counts()
    allowed_class_names = class_counts[class_counts > min_class_num].index
    csv = csv[csv['Y'].isin(allowed_class_names)].reset_index(drop=True)
    
    self.images = csv['image']
    self.targets = csv['Y']
   

  def __len__(self):
    return len(self.images)
 

  def __getitem__(self, index):
    image = cv2.cvtColor(cv2.imread(join(self.img_folder, self.images[index])), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, self.img_size)
    target = self.targets[index]
    

    if self.transform is not None:
        image = self.transform(image)
    
    return image, target


class TestImageDataset(Dataset):

  def __init__(self, img_folder, transform=None):
    self.transform = transform
    self.img_folder = img_folder
    self.images = os.listdir(img_folder)[:5]
   
  def __len__(self):
    return len(self.images)
 

  def __getitem__(self, index):
    image = cv2.cvtColor(cv2.imread(join(self.img_folder, self.images[index])), cv2.COLOR_BGR2RGB)
    
    if self.transform is not None:
        image = self.transform(image)
    
    return image, np.array([ord(char) for char in self.images[index]]).astype(np.uint8)
  

class DummyDataset(Dataset):

  def __init__(self, input_size=(256, 256), num_samples=5000, channels_num=3):
    self.input_size = input_size
    self.num_samples = num_samples
    self.channels_num = channels_num

  def __len__(self):
    return self.num_samples
 

  def __getitem__(self, index):

    image = torch.randn((self.channels_num, self.input_size[0], self.input_size[1]))
    
    return image, 5
  
class DummyDataset2(Dataset):

  def __init__(self, input_size=(256, 256), num_samples=5000, channels_num=3):
    self.input_size = input_size
    self.num_samples = num_samples
    self.channels_num = channels_num

  def __len__(self):
    return self.num_samples
 

  def __getitem__(self, index):

    image = torch.randn((self.channels_num, self.input_size[0], self.input_size[1]))
    
    return image, '123.jpg'

class DummyDataset3(Dataset):

  def __init__(self, input_size=(100, 100), num_samples=20, channels_num=4):
    self.input_size = input_size
    self.num_samples = num_samples
    self.channels_num = channels_num

  def __len__(self):
    return self.num_samples
 

  def __getitem__(self, index):

    image = np.random.rand(self.channels_num, self.input_size[0], self.input_size[1]).astype(np.float32)
    
    return image, index


class MaskImageDataset(Dataset):
  
  def __init__(self, csv, img_folder, mask_folder, img_size=(640, 640)):
    self.img_folder = img_folder
    self.img_size = img_size
    self.mask_folder = mask_folder
    self.images = csv['image']
    self.targets = csv['Y']
   

  def __len__(self):
    return len(self.images)
 

  def __getitem__(self, index):

    image = cv2.resize(cv2.cvtColor(cv2.imread(join(self.img_folder, self.images[index])), cv2.COLOR_BGR2RGB), (self.img_size[0], self.img_size[1]))
    mask = cv2.imread(join(self.img_folder, self.images[index], cv2.IMREAD_GRAYSCALE))
    target = self.targets[index]
    
    image_masked = np.concatenate((image, mask), axis=2)

    return image_masked, target