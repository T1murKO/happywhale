import imgaug.augmenters as iaa
from torchvision import transforms as T
import torch
from kornia.augmentation import *
from kornia.augmentation.container import AugmentationSequential
from torch import nn


class Augmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.aug_list = AugmentationSequential(
            RandomHorizontalFlip(p=0.5),
            AugmentationSequential(RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.2, 1.5), p=0.5),
                                   RandomMotionBlur(kernel_size=(3, 3), angle=(-90,90), direction=0, p=0.5),
                                   random_apply=1),
            ColorJitter(brightness=(0.95, 1.2),
                        contrast=(0.8, 1.3),
                        saturation=(0.7, 1.3),
                        hue=0.01,
                        p=0.6),
            AugmentationSequential(RandomThinPlateSpline(scale=0.2, p=0.65),
                                   RandomRotation(degrees=(-15, 15), p=0.65),
                                   RandomPerspective(distortion_scale=0.2, p=0.65),
                                   random_apply=1),
            
            AugmentationSequential(RandomErasing(scale=(0.02, 0.07), ratio=(0.7, 1.3), value=0.0, p=0.3),
                                   RandomErasing(scale=(0.015, 0.07), ratio=(0.7, 1.3), value=0.0, p=0.2),
                                   RandomErasing(scale=(0.015, 0.07), ratio=(0.7, 1.3), value=0.0, p=0.2),
                                   )
            
            
        )
    @torch.no_grad()
    def forward(self, x):
        return self.aug_list(x.type(torch.float32) / 255.)


def get_augmentation_list(input_size=(256, 256)):
    transform_list = T.Compose([             
        iaa.Sequential([
            iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Sometimes(0.3, iaa.AverageBlur(k=(3,5))),
            iaa.Sometimes(0.3, iaa.MotionBlur(k=(3,5))),
            iaa.Sometimes(0.3,iaa.Add((-15, 15), per_channel=0.6)),
            iaa.Sometimes(0.3, iaa.Multiply((0.8, 1.2), per_channel=0.6)),
            iaa.Sometimes(0.3, iaa.Affine(
                scale={'x': (0.85,1.2), 'y': (0.85,1.2)},
                translate_percent={'x': (-0.065,0.065), 'y': (-0.065,0.065)},
                shear=(-12,12),
                rotate=(-12,12)
                )),
            iaa.Sometimes(0.2, iaa.Grayscale(alpha=(0.7,1.0))),
            ], random_order=True),
            # iaa.size.Resize(input_size, interpolation='cubic')
        ]).augment_image,     
        T.ToTensor()
    ])
    
    return transform_list


def get_infer_list(input_size=(256, 256)):
    transforms_list_eval = T.Compose([             
    # iaa.Sequential([
    #     # iaa.size.Resize(input_size, interpolation='cubic')
    # ]).augment_image,     
    T.ToTensor()
    ])
    
    return transforms_list_eval