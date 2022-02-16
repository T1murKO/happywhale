from torch.utils.data import DataLoader, Dataset
import cv2
import torch
from os.path import join

class ImageDataset(Dataset):
  def __init__(self, csv, img_folder, transform=None):
    self.transform = transform
    self.img_folder = img_folder
     
    self.images = csv['image']
    self.targets = csv['Y']
   

  def __len__(self):
    return len(self.images)
 

  def __getitem__(self, index):

    image = cv2.cvtColor(cv2.imread(join(self.img_folder, self.images[index])), cv2.COLOR_BGR2RGB)
    target = self.targets[index]
     
    if self.transform is not None:
        image = self.transform(image)
    
    return image, target
  


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