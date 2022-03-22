# import imgaug.augmenters as iaa
from torchvision import transforms as T
import torch
from kornia.augmentation import *
from kornia.augmentation.container import AugmentationSequential
from torch import nn


class ProgressiveAugmenter(nn.Module):
    def __init__(self, aug_settings):
        super().__init__()
        self.aug_setttings = aug_settings
        self.stage_len = aug_settings['epochs_max'] // aug_settings['stages_num']
         
       
    @staticmethod
    def inrpolate_value(min, max, scale_coef):
        return min + (max - min) * scale_coef
    
    def update_augmentation_list(self, epoch):
        s = self.aug_setttings
        if epoch > self.aug_setttings['epochs_max']:
            stage = self.aug_setttings['stages_num'] - 1
        else:
            stage = epoch // self.stage_len
        
        scale_coef = stage / (self.aug_setttings['stages_num'] - 1)
        
        gblur_s_min = round(self.inrpolate_value(s['gausian_blur']['sigma'][0][0],
                                            s['gausian_blur']['sigma'][0][1],
                                            scale_coef), 3)
        
        gblur_s_max = round(self.inrpolate_value(s['gausian_blur']['sigma'][1][0],
                                            s['gausian_blur']['sigma'][1][1],
                                            scale_coef), 3)
        
        gblur_p = round(self.inrpolate_value(s['gausian_blur']['p'][0],
                                    s['gausian_blur']['p'][1],
                                    scale_coef), 3)
        
        
        mblur_p = round(self.inrpolate_value(s['motion_blur']['p'][0],
                            s['motion_blur']['p'][1],
                            scale_coef), 3)
        
        
        brightness_min = round(self.inrpolate_value(s['color_jitter']['brightness'][0][0],
                                    s['color_jitter']['brightness'][0][1],
                                    scale_coef), 3)
        
        brightness_max = round(self.inrpolate_value(s['color_jitter']['brightness'][1][0],
                            s['color_jitter']['brightness'][1][1],
                            scale_coef), 3)
        
        
        contrast_min = round(self.inrpolate_value(s['color_jitter']['contrast'][0][0],
                            s['color_jitter']['contrast'][0][1],
                            scale_coef), 3)
        
        contrast_max = round(self.inrpolate_value(s['color_jitter']['contrast'][1][0],
                            s['color_jitter']['contrast'][1][1],
                            scale_coef), 3)
        

        saturation_min = round(self.inrpolate_value(s['color_jitter']['saturation'][0][0],
                            s['color_jitter']['saturation'][0][1],
                            scale_coef), 3)
        
        saturation_max = round(self.inrpolate_value(s['color_jitter']['saturation'][1][0],
                            s['color_jitter']['saturation'][1][1],
                            scale_coef), 3)


        hue_min = round(self.inrpolate_value(s['color_jitter']['hue'][0][0],
                            s['color_jitter']['hue'][0][1],
                            scale_coef), 3)
        
        hue_max = round(self.inrpolate_value(s['color_jitter']['hue'][1][0],
                            s['color_jitter']['hue'][1][1],
                            scale_coef), 3)
        

        color_jitter_p = round(self.inrpolate_value(s['color_jitter']['p'][0],
                            s['color_jitter']['p'][1],
                            scale_coef), 3)
        
        
        th_plate_s = round(self.inrpolate_value(s['thin_plate_spline']['scale'][0],
                            s['thin_plate_spline']['scale'][1],
                            scale_coef), 3)
        
        th_plate_p = round(self.inrpolate_value(s['thin_plate_spline']['p'][0],
                            s['thin_plate_spline']['p'][1],
                            scale_coef), 3)
        
        
        rotation_degree_min = round(self.inrpolate_value(s['rotation']['degrees'][0][0],
                            s['rotation']['degrees'][0][1],
                            scale_coef), 3)
        
        rotation_degree_max = round(self.inrpolate_value(s['rotation']['degrees'][1][0],
                            s['rotation']['degrees'][1][1],
                            scale_coef), 3)
        
        rotation_p = round(self.inrpolate_value(s['rotation']['p'][0],
                            s['rotation']['p'][0],
                            scale_coef), 3)
        
        
        perspective_s = round(self.inrpolate_value(s['perspective']['distortion_scale'][0],
                            s['perspective']['distortion_scale'][1],
                            scale_coef), 3)
        
        perspective_p = round(self.inrpolate_value(s['perspective']['p'][0],
                            s['perspective']['p'][1],
                            scale_coef), 3)
        
        
        cutout1_s_min = round(self.inrpolate_value(s['cutout1']['scale'][0][0],
                            s['cutout1']['scale'][0][1],
                            scale_coef), 3)
        
        cutout1_s_max = round(self.inrpolate_value(s['cutout1']['scale'][1][0],
                            s['cutout1']['scale'][1][1],
                            scale_coef), 3)
        
        cutout1_p = round(self.inrpolate_value(s['cutout1']['p'][0],
                            s['cutout1']['p'][1],
                            scale_coef), 3)
        
        cutout2_s_min = round(self.inrpolate_value(s['cutout2']['scale'][0][0],
                    s['cutout1']['scale'][0][1],
                    scale_coef), 3)
        
        cutout2_s_max = round(self.inrpolate_value(s['cutout2']['scale'][1][0],
                            s['cutout1']['scale'][1][1],
                            scale_coef), 3)
        
        cutout2_p = round(self.inrpolate_value(s['cutout2']['p'][0],
                            s['cutout1']['p'][1],
                            scale_coef), 3)
        
        resolution = round(self.inrpolate_value(s['resolution']['min_res'],
                            s['resolution']['max_res'],
                            scale_coef))
        
        self.aug_list = AugmentationSequential(
            Resize((resolution, resolution)),
            RandomHorizontalFlip(p=0.5),
            AugmentationSequential(RandomGaussianBlur(kernel_size=(3, 3), sigma=(gblur_s_min, gblur_s_max), p=gblur_p),
                                   RandomMotionBlur(kernel_size=(3, 3), angle=(-90,90), direction=0, p=mblur_p),
                                   random_apply=1),
            ColorJitter(brightness=(brightness_min, brightness_max),
                        contrast=(contrast_min, contrast_max),
                        saturation=(saturation_min, saturation_max),
                        hue=(hue_min, hue_max),
                        p=color_jitter_p),
            

            AugmentationSequential(RandomThinPlateSpline(scale=th_plate_s, p=th_plate_p),
                                   RandomRotation(degrees=(rotation_degree_min, rotation_degree_max), p=rotation_p),
                                   RandomPerspective(distortion_scale=perspective_s, p=perspective_p),
                                   random_apply=1),
            
            AugmentationSequential(RandomErasing(scale=(cutout1_s_min, cutout1_s_max), ratio=(0.7, 1.3), value=0.0, p=cutout1_p),
                                   RandomErasing(scale=(cutout2_s_min, cutout2_s_max), ratio=(0.7, 1.3), value=0.0, p=cutout2_p),
                                   )
        )
        return resolution
        
    @torch.no_grad()
    def forward(self, x):
        return self.aug_list(x / 255.)


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