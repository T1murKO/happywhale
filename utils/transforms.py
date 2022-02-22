import imgaug.augmenters as iaa
from torchvision import transforms as T


def crop_box(img, box, exntend_share=(0.15, 1)):
    pass


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