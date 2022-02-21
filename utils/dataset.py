from torch.utils.data import DataLoader, Dataset
import cv2
import torch
from os.path import join
import os
from transforms import crop_box

class TrainImageDataset(Dataset):

  def __init__(self, csv, img_folder, transform=None, box_csv=None):
    self.transform = transform
    self.img_folder = img_folder
     
    self.images = csv['image']
    self.targets = csv['Y']
    self.box_csv = box_csv
    self.box_csv = self.box_csv['box'].apply(lambda x: [float(i) for i in x.split()])
   

  def __len__(self):
    return len(self.images)
 

  def __getitem__(self, index):

    image = cv2.cvtColor(cv2.imread(join(self.img_folder, self.images[index])), cv2.COLOR_BGR2RGB)
    target = self.targets[index]
    

    if self.transform is not None:
        if self.box_csv is not None:
          image = crop_box(image, self.box_csv[index])
          
        image = self.transform(image)
    
    return image, target


class TestImageDataset(Dataset):

  def __init__(self, img_folder, transform=None):
    self.transform = transform
    self.img_folder = img_folder
    self.images = os.listdir(img_folder)
   
  def __len__(self):
    return len(self.images)
 

  def __getitem__(self, index):
    image = cv2.cvtColor(cv2.imread(join(self.img_folder, self.images[index])), cv2.COLOR_BGR2RGB)
    
    if self.transform is not None:
        image = self.transform(image)
    
    return image, self.images[index]
  

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


class MaskImageDataset(Dataset):
  
  def __init__(self, csv, img_folder, transform=None, img_size=(512, 512)):
    self.transform = transform
    self.img_folder = img_folder
    self.img_size = img_size
     
    self.images = csv['image']
    self.targets = csv['Y']
   

  def __len__(self):
    return len(self.images)
 

  def __getitem__(self, index):

    image = cv2.cvtColor(cv2.imread(join(self.img_folder, self.images[index])), cv2.COLOR_BGR2RGB)
    target = self.targets[index]
     
    if self.transform is not None:
        image = self.transform(image)
    
    # print(image.shape)
    mask = torch.randn(1, self.img_size[0], self.img_size[1])
    image_mask = torch.cat((image, mask), axis=0)

    return image_mask, target