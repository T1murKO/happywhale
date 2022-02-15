from torch.utils.data import DataLoader, Dataset
import cv2
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